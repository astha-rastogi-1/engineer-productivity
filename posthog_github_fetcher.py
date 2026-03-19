import argparse
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests


REPO_URL_DEFAULT = "https://github.com/PostHog/posthog"


def _parse_repo_owner_and_name(repo_url: str) -> Tuple[str, str]:
    """
    Accepts a URL like:
      - https://github.com/PostHog/posthog
      - github.com/PostHog/posthog
    and returns (owner, repo).
    """
    # If the user passes "github.com/OWNER/REPO" without a scheme, add a scheme for parsing.
    if "://" not in repo_url:
        repo_url = "https://" + repo_url
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Could not extract owner/repo from repo_url={repo_url!r}")
    return parts[0], parts[1]


def _to_utc_iso_z(dt: datetime) -> str:
    """Store timestamps in ISO-8601 UTC with a trailing 'Z' (lexicographically sortable)."""
    dt_utc = dt.astimezone(timezone.utc)
    # Example: 2026-03-19T12:34:56Z
    return dt_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_github_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    # GitHub typically returns ISO strings like: "2020-01-01T12:34:56Z"
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


def _safe_iso_z(dt_str: Optional[str]) -> Optional[str]:
    """Convert GitHub ISO datetime strings to our UTC ISO-8601 '...Z' representation."""
    parsed = _parse_github_datetime(dt_str)
    if parsed is None:
        return None
    return _to_utc_iso_z(parsed)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _load_token() -> str:
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token

    # Fallback to local config.py if the user already has it (repo currently contains one).
    # Note: This is only a fallback; prefer setting `GITHUB_TOKEN` in your environment.
    try:
        import config  # type: ignore

        cfg_token = getattr(config, "github_access_token", None)
        if cfg_token:
            return str(cfg_token)
    except Exception:
        pass

    raise RuntimeError(
        "Missing GitHub token. Set `GITHUB_TOKEN` env var, or update `config.py` with `github_access_token`."
    )


@dataclass
class RateLimitInfo(Exception):
    reset_epoch: int
    wait_seconds: int

    def __str__(self) -> str:
        reset_iso = datetime.fromtimestamp(self.reset_epoch, tz=timezone.utc).isoformat()
        return f"Rate limited; reset at {reset_iso}, waiting {self.wait_seconds}s"


def _compute_wait_seconds(reset_epoch: Optional[int]) -> int:
    if not reset_epoch:
        return 60
    # Add a small buffer so we retry after the reset moment.
    return max(0, int(reset_epoch - time.time()) + 5)


def _request_json_with_retries(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    *,
    max_retries: int = 6,
    timeout_seconds: int = 30,
) -> Any:
    """
    Make a GET request and return JSON.

    Retries:
      - Transient 5xx / network issues: exponential backoff.
      - Rate limit 403: raise RateLimitInfo with reset time so the caller can sleep+resume.
    """
    backoff_seconds = 1.0
    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout_seconds)

            # Handle explicit GitHub rate limiting.
            if resp.status_code in {403, 429}:
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset = resp.headers.get("X-RateLimit-Reset")
                reset_epoch = int(reset) if reset and reset.isdigit() else None
                remaining_zero = remaining == "0"

                # Some 403 responses include "API rate limit exceeded" in JSON message.
                message = ""
                try:
                    payload = resp.json()
                    message = str(payload.get("message", ""))
                except Exception:
                    payload = None
                    message = ""

                retry_after = resp.headers.get("Retry-After")
                is_rate_limited = remaining_zero or ("rate limit" in message.lower())

                # For 429 we generally want to treat it as rate limiting even if message varies.
                if resp.status_code == 429:
                    is_rate_limited = True

                if is_rate_limited:
                    wait_seconds = (
                        int(retry_after)
                        if retry_after and retry_after.isdigit()
                        else _compute_wait_seconds(reset_epoch)
                    )
                    raise RateLimitInfo(
                        reset_epoch=reset_epoch or int(time.time()) + wait_seconds,
                        wait_seconds=wait_seconds,
                    )

            # Retry common transient server errors.
            if resp.status_code in {502, 503, 504}:
                resp.raise_for_status()

            # For other 4xx (besides rate limits), don't retry.
            if 400 <= resp.status_code < 500:
                resp.raise_for_status()

            resp.raise_for_status()
            return resp.json()
        except RateLimitInfo:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            # Exponential backoff with jitter-ish behavior via attempt.
            time.sleep(backoff_seconds)
            backoff_seconds *= 2.0 * min(1.5, attempt / 2.0)

    assert last_exc is not None
    raise last_exc


def _ensure_db_schema(conn: sqlite3.Connection) -> None:
    # NOTE: We use schema migration logic because the table may already exist from
    # previous runs (e.g. when adding new columns like comments_count).
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pull_requests (
            id INTEGER PRIMARY KEY,
            number INTEGER NOT NULL,
            title TEXT,
            state TEXT,
            created_at TEXT,
            updated_at TEXT,
            closed_at TEXT,
            merged_at TEXT,
            author_login TEXT,
            html_url TEXT,
            base_ref TEXT,
            head_ref TEXT,
            additions INTEGER,
            deletions INTEGER,
            changed_files INTEGER,
            comments_count INTEGER
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS issues (
            id INTEGER PRIMARY KEY,
            number INTEGER NOT NULL,
            title TEXT,
            state TEXT,
            created_at TEXT,
            updated_at TEXT,
            closed_at TEXT,
            author_login TEXT,
            html_url TEXT,
            labels_json TEXT,
            assignees_json TEXT,
            comments_count INTEGER
        );
        """
    )
    conn.commit()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set:
    cur = conn.execute(f"PRAGMA table_info({table_name});")
    return {row[1] for row in cur.fetchall()}  # row[1] is column name


def _ensure_additional_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    """
    Add missing columns for existing tables.

    `columns` is a mapping: column_name -> sqlite_type (e.g. "INTEGER").
    """
    existing = _table_columns(conn, table)
    for col, col_type in columns.items():
        if col in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type};")
    conn.commit()


def _load_seen_ids(conn: sqlite3.Connection, table_name: str) -> set:
    cur = conn.execute(f"SELECT id FROM {table_name}")
    return {row[0] for row in cur.fetchall()}


def _upsert_pr_rows(conn: sqlite3.Connection, rows: List[Tuple[Any, ...]]) -> None:
    # Upsert so we can backfill missing additions/comments_count in later runs.
    conn.executemany(
        """
        INSERT INTO pull_requests (
            id, number, title, state, created_at, updated_at, closed_at, merged_at,
            author_login, html_url, base_ref, head_ref, additions, deletions, changed_files, comments_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            number=excluded.number,
            title=excluded.title,
            state=excluded.state,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at,
            closed_at=excluded.closed_at,
            merged_at=excluded.merged_at,
            author_login=excluded.author_login,
            html_url=excluded.html_url,
            base_ref=excluded.base_ref,
            head_ref=excluded.head_ref,
            additions=excluded.additions,
            deletions=excluded.deletions,
            changed_files=excluded.changed_files,
            comments_count=excluded.comments_count;
        """,
        rows,
    )


def _upsert_issue_rows(conn: sqlite3.Connection, rows: List[Tuple[Any, ...]]) -> None:
    conn.executemany(
        """
        INSERT INTO issues (
            id, number, title, state, created_at, updated_at, closed_at,
            author_login, html_url, labels_json, assignees_json, comments_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            number=excluded.number,
            title=excluded.title,
            state=excluded.state,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at,
            closed_at=excluded.closed_at,
            author_login=excluded.author_login,
            html_url=excluded.html_url,
            labels_json=excluded.labels_json,
            assignees_json=excluded.assignees_json,
            comments_count=excluded.comments_count;
        """,
        rows,
    )


def _extract_labels(issue_obj: Dict[str, Any]) -> List[str]:
    labels = issue_obj.get("labels") or []
    return [lbl.get("name") for lbl in labels if isinstance(lbl, dict) and lbl.get("name")]


def _extract_assignees(issue_obj: Dict[str, Any]) -> List[str]:
    assignees = issue_obj.get("assignees") or []
    return [a.get("login") for a in assignees if isinstance(a, dict) and a.get("login")]


def _build_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            # Helps ensure newer API behavior; harmless if unused.
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "posthog-github-fetcher",
        }
    )
    return s


def _save_state(state_path: str, payload: Dict[str, Any]) -> None:
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, state_path)


def _load_state(state_path: str) -> Dict[str, Any]:
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_posthog_prs_and_issues(
    *,
    repo_url: str,
    days: int,
    db_path: str,
    state_path: str,
    output_dir: str,
    per_page: int,
    max_pages: int,
    resume: bool,
) -> None:
    owner, repo = _parse_repo_owner_and_name(repo_url)
    token = _load_token()
    session = _build_session(token)

    os.makedirs(output_dir, exist_ok=True)
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = _to_utc_iso_z(cutoff_dt)

    conn = sqlite3.connect(db_path)
    try:
        _ensure_db_schema(conn)
        _ensure_additional_columns(
            conn,
            "pull_requests",
            {"comments_count": "INTEGER"},
        )
        _ensure_additional_columns(
            conn,
            "issues",
            {"comments_count": "INTEGER"},
        )

        # Load state (if resuming) so we can continue pagination without re-inserting.
        state = _load_state(state_path) if resume else {}
        if state.get("repo_url") != repo_url or state.get("cutoff_iso") != cutoff_iso:
            state = {}

        pr_page = int(state.get("pr_page", 1))
        issue_page = int(state.get("issue_page", 1))

        # Load "seen IDs" from the database so we do not insert duplicates.
        seen_pr_ids = _load_seen_ids(conn, "pull_requests")
        seen_issue_ids = _load_seen_ids(conn, "issues")

        prs_all_new_records_count = 0
        issues_all_new_records_count = 0

        def _fetch_pr_detail(pull_number: int) -> Dict[str, Any]:
            # The "list PRs" endpoint may omit additions/deletions/changed_files.
            # The per-PR endpoint is the reliable source.
            detail_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
            return _request_json_with_retries(session, detail_url, params={})

        def _fetch_issue_detail(issue_number: int) -> Dict[str, Any]:
            detail_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
            return _request_json_with_retries(session, detail_url, params={})

        # --- Pull Requests ---
        pulls_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        pr_params_base = {
            "state": "all",
            "sort": "updated",
            "direction": "desc",
            "per_page": per_page,
        }

        for page in range(pr_page, max_pages + 1):
            pr_params = dict(pr_params_base)
            pr_params["page"] = page

            try:
                pr_list = _request_json_with_retries(session, pulls_url, pr_params)
            except RateLimitInfo as e:
                _save_state(
                    state_path,
                    {
                        "repo_url": repo_url,
                        "cutoff_iso": cutoff_iso,
                        "pr_page": page,
                        "issue_page": issue_page,
                        "rate_limited_at": int(time.time()),
                        "reset_epoch": e.reset_epoch,
                        "wait_seconds": e.wait_seconds,
                    },
                )
                time.sleep(e.wait_seconds)
                # Reload to honor "skip already added" even if this run previously inserted some PRs.
                seen_pr_ids = _load_seen_ids(conn, "pull_requests")
                continue

            if not pr_list:
                # No more PRs.
                pr_page = page + 1
                break

            page_stop_by_cutoff = False

            prs_page_rows: List[Tuple[Any, ...]] = []

            for pr in pr_list:
                pr_id = pr.get("id")
                pr_updated = _parse_github_datetime(pr.get("updated_at"))
                if pr_id is None or pr_updated is None:
                    continue

                pr_updated_iso = _to_utc_iso_z(pr_updated)
                if pr_updated_iso < cutoff_iso:
                    # Because the endpoint is sorted by updated desc, once we're past cutoff we can stop.
                    page_stop_by_cutoff = True
                    break

                if pr_id in seen_pr_ids:
                    continue

                user = pr.get("user") or {}
                base = (pr.get("base") or {}).get("ref")
                head = (pr.get("head") or {}).get("ref")

                # Enrich missing numeric fields from the per-PR endpoint.
                pr_detail: Optional[Dict[str, Any]] = None
                if pr.get("additions") is None or pr.get("deletions") is None or pr.get("changed_files") is None:
                    pr_detail = _fetch_pr_detail(int(pr.get("number")))

                additions = pr_detail.get("additions") if pr_detail else pr.get("additions")
                deletions = pr_detail.get("deletions") if pr_detail else pr.get("deletions")
                changed_files = pr_detail.get("changed_files") if pr_detail else pr.get("changed_files")

                # GitHub returns:
                #   - `comments`: non-review comments on the PR
                #   - `review_comments`: inline comments on the PR
                review_comments = (pr_detail.get("review_comments") if pr_detail else pr.get("review_comments")) or 0
                comments = (pr_detail.get("comments") if pr_detail else pr.get("comments")) or 0
                comments_count = int(comments) + int(review_comments)

                prs_page_rows.append(
                    (
                        pr_id,
                        pr.get("number"),
                        pr.get("title"),
                        pr.get("state"),
                        _safe_iso_z(pr.get("created_at")),
                        pr_updated_iso,
                        _safe_iso_z(pr.get("closed_at")),
                        _safe_iso_z(pr.get("merged_at")),
                        user.get("login"),
                        pr.get("html_url"),
                        base,
                        head,
                        additions,
                        deletions,
                        changed_files,
                        comments_count,
                    )
                )

            if prs_page_rows:
                _upsert_pr_rows(conn, prs_page_rows)
                conn.commit()
                prs_all_new_records_count += len(prs_page_rows)
                for r in prs_page_rows:
                    seen_pr_ids.add(r[0])

            pr_page = page + 1
            _save_state(
                state_path,
                {
                    "repo_url": repo_url,
                    "cutoff_iso": cutoff_iso,
                    "pr_page": pr_page,
                    "issue_page": issue_page,
                    "last_pr_page_processed": page,
                    "prs_new_in_page": len(prs_page_rows),
                    "cutoff_iso": cutoff_iso,
                },
            )

            if page_stop_by_cutoff:
                break

        # --- Issues (excluding PRs) ---
        issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        issue_params_base = {
            "state": "all",
            "since": cutoff_iso,
            "per_page": per_page,
        }

        for page in range(issue_page, max_pages + 1):
            issue_params = dict(issue_params_base)
            issue_params["page"] = page

            try:
                issue_list = _request_json_with_retries(session, issues_url, issue_params)
            except RateLimitInfo as e:
                _save_state(
                    state_path,
                    {
                        "repo_url": repo_url,
                        "cutoff_iso": cutoff_iso,
                        "pr_page": pr_page,
                        "issue_page": page,
                        "rate_limited_at": int(time.time()),
                        "reset_epoch": e.reset_epoch,
                        "wait_seconds": e.wait_seconds,
                    },
                )
                time.sleep(e.wait_seconds)
                seen_issue_ids = _load_seen_ids(conn, "issues")
                continue

            if not issue_list:
                issue_page = page + 1
                break

            issues_page_rows: List[Tuple[Any, ...]] = []

            for issue in issue_list:
                # /issues returns both issues and PRs; we want issues only.
                if issue.get("pull_request") is not None:
                    continue

                issue_id = issue.get("id")
                issue_updated = _parse_github_datetime(issue.get("updated_at"))
                if issue_id is None or issue_updated is None:
                    continue

                issue_updated_iso = _to_utc_iso_z(issue_updated)
                if issue_updated_iso < cutoff_iso:
                    # With `since`, this should mostly be redundant, but keep it as a safety net.
                    continue

                if issue_id in seen_issue_ids:
                    continue

                user = issue.get("user") or {}
                issues_page_rows.append(
                    (
                        issue_id,
                        issue.get("number"),
                        issue.get("title"),
                        issue.get("state"),
                        _safe_iso_z(issue.get("created_at")),
                        issue_updated_iso,
                        _safe_iso_z(issue.get("closed_at")),
                        user.get("login"),
                        issue.get("html_url"),
                        _json_dumps(_extract_labels(issue)),
                        _json_dumps(_extract_assignees(issue)),
                        issue.get("comments"),
                    )
                )

            if issues_page_rows:
                _upsert_issue_rows(conn, issues_page_rows)
                conn.commit()
                issues_all_new_records_count += len(issues_page_rows)
                for r in issues_page_rows:
                    seen_issue_ids.add(r[0])

            issue_page = page + 1
            _save_state(
                state_path,
                {
                    "repo_url": repo_url,
                    "cutoff_iso": cutoff_iso,
                    "pr_page": pr_page,
                    "issue_page": issue_page,
                    "last_issue_page_processed": page,
                    "issues_new_in_page": len(issues_page_rows),
                    "cutoff_iso": cutoff_iso,
                },
            )

        # --- Final DataFrames + CSV exports ---
        # Backfill missing numeric fields/comments from per-item endpoints.
        # This ensures older rows that were previously stored with NULLs get fixed.
        missing_prs_cur = conn.execute(
            """
            SELECT id, number
            FROM pull_requests
            WHERE (additions IS NULL OR deletions IS NULL OR changed_files IS NULL OR comments_count IS NULL)
              AND updated_at >= ?
            ORDER BY updated_at DESC, id DESC;
            """,
            (cutoff_iso,),
        )
        missing_prs = missing_prs_cur.fetchall()
        for (pid, pnumber) in missing_prs:
            try:
                detail = _fetch_pr_detail(int(pnumber))
            except RateLimitInfo as e:
                _save_state(
                    state_path,
                    {
                        "repo_url": repo_url,
                        "cutoff_iso": cutoff_iso,
                        "pr_page": pr_page,
                        "issue_page": issue_page,
                        "rate_limited_at": int(time.time()),
                        "reset_epoch": e.reset_epoch,
                        "wait_seconds": e.wait_seconds,
                    },
                )
                time.sleep(e.wait_seconds)
                detail = _fetch_pr_detail(int(pnumber))

            additions = detail.get("additions")
            deletions = detail.get("deletions")
            changed_files = detail.get("changed_files")
            comments_count = int(detail.get("comments") or 0) + int(detail.get("review_comments") or 0)

            conn.execute(
                """
                UPDATE pull_requests
                SET additions = ?, deletions = ?, changed_files = ?, comments_count = ?
                WHERE id = ?;
                """,
                (additions, deletions, changed_files, comments_count, pid),
            )
        conn.commit()

        missing_issues_cur = conn.execute(
            """
            SELECT id, number
            FROM issues
            WHERE comments_count IS NULL
              AND updated_at >= ?
            ORDER BY updated_at DESC, id DESC;
            """,
            (cutoff_iso,),
        )
        missing_issues = missing_issues_cur.fetchall()
        for (iid, inumber) in missing_issues:
            try:
                detail = _fetch_issue_detail(int(inumber))
            except RateLimitInfo as e:
                _save_state(
                    state_path,
                    {
                        "repo_url": repo_url,
                        "cutoff_iso": cutoff_iso,
                        "pr_page": pr_page,
                        "issue_page": issue_page,
                        "rate_limited_at": int(time.time()),
                        "reset_epoch": e.reset_epoch,
                        "wait_seconds": e.wait_seconds,
                    },
                )
                time.sleep(e.wait_seconds)
                detail = _fetch_issue_detail(int(inumber))

            comments_count = detail.get("comments")
            conn.execute(
                """
                UPDATE issues
                SET comments_count = ?
                WHERE id = ?;
                """,
                (comments_count, iid),
            )
        conn.commit()

        # Export *all* rows currently in the DB within the window, not only the new ones.
        prs_df = pd.read_sql_query(
            """
            SELECT
                'pull_request' AS item_type,
                id, number, title, state,
                created_at, updated_at, closed_at, merged_at,
                author_login, html_url,
                base_ref, head_ref,
                additions, deletions, changed_files
                , comments_count
            FROM pull_requests
            WHERE updated_at >= ?
            ORDER BY updated_at DESC, id DESC;
            """,
            conn,
            params=[cutoff_iso],
        )
        issues_df = pd.read_sql_query(
            """
            SELECT
                'issue' AS item_type,
                id, number, title, state,
                created_at, updated_at, closed_at,
                author_login, html_url,
                NULL AS merged_at,
                NULL AS base_ref,
                NULL AS head_ref,
                NULL AS additions,
                NULL AS deletions,
                NULL AS changed_files,
                labels_json,
                assignees_json,
                comments_count
            FROM issues
            WHERE updated_at >= ?
            ORDER BY updated_at DESC, id DESC;
            """,
            conn,
            params=[cutoff_iso],
        )

        # Ensure both outputs share a consistent set of columns.
        # (We'll keep labels/assignees only on the issues side.)
        all_df = pd.concat([prs_df, issues_df], ignore_index=True, sort=False)

        prs_csv = os.path.join(output_dir, f"{owner}_{repo}_pull_requests_last_{days}_days.csv")
        issues_csv = os.path.join(output_dir, f"{owner}_{repo}_issues_last_{days}_days.csv")
        all_csv = os.path.join(output_dir, f"{owner}_{repo}_prs_and_issues_last_{days}_days.csv")

        prs_df.to_csv(prs_csv, index=False)
        issues_df.to_csv(issues_csv, index=False)
        all_df.to_csv(all_csv, index=False)

        print(f"Done. Window starts at {cutoff_iso}")
        print(f"New PRs inserted this run: {prs_all_new_records_count}")
        print(f"New issues inserted this run: {issues_all_new_records_count}")
        print(f"Wrote: {prs_csv}")
        print(f"Wrote: {issues_csv}")
        print(f"Wrote: {all_csv}")
    finally:
        conn.close()


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_db_path = os.path.join(script_dir, "data", "posthog_github.sqlite3")
    default_state_path = os.path.join(script_dir, "data", "fetch_state.json")
    default_output_dir = os.path.join(script_dir, "output")

    parser = argparse.ArgumentParser(description="Fetch PostHog PRs and Issues for the last N days.")
    parser.add_argument("--repo-url", default=REPO_URL_DEFAULT, help="GitHub repo URL (e.g. https://github.com/PostHog/posthog)")
    parser.add_argument("--days", type=int, default=90, help="Lookback window in days (default: 90)")
    parser.add_argument("--db-path", default=default_db_path, help="SQLite DB path for checkpointing/dedup")
    parser.add_argument("--state-path", default=default_state_path, help="JSON state file for resumable pagination")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory to write CSV outputs")
    parser.add_argument("--per-page", type=int, default=100, help="API page size (default: 100)")
    parser.add_argument("--max-pages", type=int, default=200, help="Safety cap on pagination loops (default: 200)")
    parser.add_argument("--no-resume", action="store_true", help="Disable loading checkpoint state from disk.")
    args = parser.parse_args()

    fetch_posthog_prs_and_issues(
        repo_url=args.repo_url,
        days=args.days,
        db_path=args.db_path,
        state_path=args.state_path,
        output_dir=args.output_dir,
        per_page=args.per_page,
        max_pages=args.max_pages,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

