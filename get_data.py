#!/usr/bin/env python3
"""
Fetch GitHub Pull Requests from the last 90 days using the GitHub REST API.

Fetches (per PR):
  - author
  - created_at, merged_at
  - additions, deletions, changed_files
  - comments (issue comments)
  - review_comments (review comments)

Results are stored in a pandas DataFrame.

Usage:
  export GITHUB_TOKEN="..."
  python test.py
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger("github_pr_fetcher")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_session(token: str) -> requests.Session:
    """Create a requests session with retry/backoff for transient failures."""
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "pr-fetcher/1.0",
        }
    )

    retry_cfg = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,  # We'll raise explicitly after custom handling.
    )
    adapter = HTTPAdapter(max_retries=retry_cfg)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _parse_next_link(link_header: Optional[str]) -> Optional[str]:
    """
    Parse GitHub-style Link header to extract the `rel="next"` URL.
    Example: <https://api.github.com/...&page=2>; rel="next", <...>; rel="last"
    """
    if not link_header:
        return None
    parts = [p.strip() for p in link_header.split(",")]
    for part in parts:
        if 'rel="next"' in part:
            start = part.find("<")
            end = part.find(">")
            if start != -1 and end != -1:
                return part[start + 1 : end]
    return None


def _safe_optional_int(value: Any) -> Optional[int]:
    """
    Convert to int when possible; return None when the API value is missing.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_json_with_rate_limit_handling(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_s: int = 30,
    max_manual_wait_s: int = 300,
) -> requests.Response:
    """
    GET with explicit handling for hard rate-limit situations.

    Note: urllib3 Retry will retry many transient errors; for GitHub rate limits,
    we may need to wait until reset time.
    """
    while True:
        resp = session.get(url, params=params, timeout=timeout_s)

        if resp.status_code in (403, 429):
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            retry_after = resp.headers.get("Retry-After")

            wait_s: Optional[int] = None
            if retry_after and retry_after.isdigit():
                wait_s = int(retry_after)
            elif remaining == "0" and reset and reset.isdigit():
                # X-RateLimit-Reset is epoch seconds.
                wait_s = max(0, int(reset) - int(time.time()) + 1)

            if wait_s is not None and wait_s > 0:
                wait_s = min(wait_s, max_manual_wait_s)
                logger.warning(
                    "GitHub rate limit encountered (HTTP %s). Sleeping %s seconds before retrying.",
                    resp.status_code,
                    wait_s,
                )
                time.sleep(wait_s)
                continue

        resp.raise_for_status()
        return resp


@dataclass(frozen=True)
class PRRow:
    author: Optional[str]
    base_ref: Optional[str]
    created_at: Optional[str]
    merged_at: Optional[str]
    additions: Optional[int]
    deletions: Optional[int]
    changed_files: Optional[int]
    comments: Optional[int]
    review_comments: Optional[int]


def _normalize_pr_object(pr: Dict[str, Any]) -> PRRow:
    """
    Normalize PR fields and ensure required columns are always populated.

    - `comments`: issue comments count
    - `review_comments`: review comments count
    """
    author_login = (pr.get("user") or {}).get("login")
    base_ref = None
    base_obj = pr.get("base")
    if isinstance(base_obj, dict):
        base_ref = base_obj.get("ref")

    return PRRow(
        author=author_login,
        base_ref=base_ref,
        created_at=pr.get("created_at"),
        merged_at=pr.get("merged_at"),
        additions=_safe_optional_int(pr.get("additions")),
        deletions=_safe_optional_int(pr.get("deletions")),
        changed_files=_safe_optional_int(pr.get("changed_files")),
        comments=_safe_optional_int(pr.get("comments")),
        review_comments=_safe_optional_int(pr.get("review_comments")),
    )


def fetch_pull_requests_last_n_days(
    owner: str,
    repo: str,
    last_days: int = 90,
    *,
    token: Optional[str] = None,
    per_page: int = 100,
) -> pd.DataFrame:
    """
    Fetch PRs from the last `last_days` days, handling pagination and retries.

    Always returns a DataFrame with:
      author, created_at, merged_at, additions, deletions, changed_files, comments, review_comments

    This implementation does *not* do per-PR extra API calls; therefore if the PR list
    endpoint does not include a field, its value will be returned as null (`None` / `pd.NA`).
    """
    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Missing GitHub token. Set `GITHUB_TOKEN` env var or pass `token=` explicitly.")

    session = _build_session(token)
    cutoff_utc = datetime.now(timezone.utc) - timedelta(days=last_days)

    pr_list_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

    logger.info(
        "Fetching PRs: %s/%s from last %s days (cutoff=%s)",
        owner,
        repo,
        last_days,
        cutoff_utc.isoformat(),
    )

    rows: List[PRRow] = []
    page = 1
    reached_cutoff = False
    total_seen = 0
    total_included = 0

    # Track whether the PR list payload includes each field at all.
    # If a field is never present in list responses, we consider it "not possible" without
    # calling the PR detail endpoint.
    pr_list_field_presence: Dict[str, bool] = {
        "created_at": False,
        "merged_at": False,
        "additions": False,
        "deletions": False,
        "changed_files": False,
        "comments": False,
        "review_comments": False,
    }
    author_field_present = False

    while not reached_cutoff:
        params: Dict[str, Any] = {
            "state": "all",
            "sort": "created",
            "direction": "desc",
            "per_page": per_page,
            "page": page,
        }

        logger.info("Requesting page=%s (per_page=%s)", page, per_page)
        resp = _get_json_with_rate_limit_handling(session, pr_list_url, params=params)
        pr_list: List[Dict[str, Any]] = resp.json()

        if not pr_list:
            logger.info("No PRs returned for page=%s; stopping.", page)
            break

        total_seen += len(pr_list)

        # Since we request sort=created&direction=desc, created_at DESC, we can stop
        # as soon as we encounter PRs older than cutoff.
        for pr in pr_list:
            # Update payload presence flags (presence, not just non-null values).
            for k in pr_list_field_presence.keys():
                if k in pr:
                    pr_list_field_presence[k] = True
            user_obj = pr.get("user")
            if isinstance(user_obj, dict) and user_obj.get("login") is not None:
                author_field_present = True

            created_at = pr.get("created_at")
            created_dt = pd.to_datetime(created_at, utc=True, errors="coerce") if created_at else pd.NaT
            if pd.isna(created_dt):
                # created_at is required for cutoff logic; skip malformed records.
                continue

            if created_dt < cutoff_utc:
                reached_cutoff = True
                break

            rows.append(_normalize_pr_object(pr))
            total_included += 1

        # Pagination: prefer Link header next URL; otherwise fall back to page increment.
        next_url = _parse_next_link(resp.headers.get("Link"))
        if next_url:
            parsed = urlparse(next_url)
            qs = parse_qs(parsed.query)
            next_pages = qs.get("page", [])
            page = int(next_pages[0]) if next_pages else page + 1
        else:
            page += 1

        logger.info(
            "Progress: included=%s (seen=%s). reached_cutoff=%s",
            total_included,
            total_seen,
            reached_cutoff,
        )

    logger.info("Done. Total included PRs in last %s days: %s", last_days, total_included)

    # Report which fields couldn't be fetched from the PR list endpoint alone.
    not_possible_fields: List[str] = []
    if not author_field_present:
        not_possible_fields.append("author")
    for api_key, present in pr_list_field_presence.items():
        if not present:
            not_possible_fields.append(api_key)

    if not_possible_fields:
        logger.warning(
            "Fields not available from PR list endpoint (no per-PR calls made): %s",
            not_possible_fields,
        )
    else:
        logger.info("All requested fields were present in PR list endpoint responses.")

    # Also print for visibility in non-log contexts (always, even if empty).
    print(f"Fields not possible without extra PR calls: {not_possible_fields}")

    df = pd.DataFrame(
        [
            {
                "author": r.author,
                "base_ref": r.base_ref,
                "created_at": r.created_at,
                "merged_at": r.merged_at,
                "additions": r.additions,
                "deletions": r.deletions,
                "changed_files": r.changed_files,
                "comments": r.comments,
                "review_comments": r.review_comments,
            }
            for r in rows
        ],
        columns=[
            "author",
            "base_ref",
            "created_at",
            "merged_at",
            "additions",
            "deletions",
            "changed_files",
            "comments",
            "review_comments",
        ],
    )

    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["merged_at"] = pd.to_datetime(df["merged_at"], utc=True, errors="coerce")
    for col in ["additions", "deletions", "changed_files", "comments", "review_comments"]:
        # Keep missing/unavailable values as nulls.
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df.sort_values("created_at", ascending=False).reset_index(drop=True)


def compute_user_features(pr_df: pd.DataFrame, *, recent_days: int = 30) -> pd.DataFrame:
    """
    Compute per-user PR features.

    Expected (recommended) input columns:
      - author or author_login
      - created_at
      - merged_at (null if not merged)
      - base_ref (base branch ref name for product impact ratio)
    """
    if pr_df.empty:
        return pd.DataFrame(
            columns=[
                "user",
                "total_prs_created",
                "total_prs_merged",
                "merge_rate",
                "median_time_to_merge_days",
                "stability_time_to_merge_variance_days",
                "days_active",
                "recent_prs",
                "product_impact_ratio",
            ]
        )

    df = pr_df.copy()

    user_col = "author" if "author" in df.columns else ("author_login" if "author_login" in df.columns else None)
    if user_col is None:
        raise ValueError("Input DataFrame must have either `author` or `author_login` column.")

    # Parse/normalize time columns.
    df["created_at"] = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")
    df["merged_at"] = pd.to_datetime(df.get("merged_at"), utc=True, errors="coerce")

    created_mask = df["created_at"].notna()
    merged_mask = df["merged_at"].notna()

    now_utc = datetime.now(timezone.utc)
    recent_cutoff = now_utc - timedelta(days=recent_days)

    # Time-to-merge (in days). Keep null for PRs that are not merged or have missing timestamps.
    time_to_merge_days = (df["merged_at"] - df["created_at"]).dt.total_seconds() / 86400.0
    time_to_merge_days = time_to_merge_days.where(merged_mask & created_mask)
    time_to_merge_days = time_to_merge_days.where(time_to_merge_days >= 0)  # drop any unexpected negatives
    df["time_to_merge_days"] = time_to_merge_days

    # Total created/merged.
    total_created = df.loc[created_mask].groupby(user_col).size().rename("total_prs_created")
    total_merged = df.loc[merged_mask].groupby(user_col).size().rename("total_prs_merged")

    # Merge rate.
    merge_rate = (total_merged / total_created).rename("merge_rate")

    # Median time to merge + stability (variance).
    median_ttm = df.groupby(user_col)["time_to_merge_days"].median().rename("median_time_to_merge_days")
    stability = df.groupby(user_col)["time_to_merge_days"].var(ddof=0).rename("stability_time_to_merge_variance_days")

    # Days active: unique created calendar days (UTC).
    activity_day = df.loc[created_mask, "created_at"].dt.normalize()
    days_active = activity_day.groupby(df.loc[created_mask, user_col]).nunique().rename("days_active")

    # Recent PRs: created in the last `recent_days`.
    recent_mask = created_mask & (df["created_at"] >= recent_cutoff)
    recent_prs = df.loc[recent_mask].groupby(user_col).size().rename("recent_prs")

    # Product impact ratio: fraction merged PRs? User asked fraction where base_ref == "main" (among all PRs).
    if "base_ref" in df.columns:
        main_mask = df["base_ref"].eq("main") & created_mask
        main_count = df.loc[main_mask].groupby(user_col).size()
        product_impact_ratio = (main_count / total_created).rename("product_impact_ratio")
    else:
        product_impact_ratio = pd.Series(dtype="float64", name="product_impact_ratio")

    result = pd.concat(
        [
            total_created,
            total_merged,
            merge_rate,
            median_ttm,
            stability,
            days_active,
            recent_prs,
            product_impact_ratio,
        ],
        axis=1,
    ).reset_index()

    result = result.rename(columns={user_col: "user"})

    # If base_ref was missing, fill null column with pd.NA.
    if "product_impact_ratio" in result.columns and result["product_impact_ratio"].isna().all():
        result["product_impact_ratio"] = pd.NA

    return result.sort_values("total_prs_created", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    df = fetch_pull_requests_last_n_days(owner="PostHog", repo="posthog", last_days=90)
    logger.info("DataFrame shape: %s", df.shape)
    print(df.head().to_string(index=False))

    # Persist results to CSV for downstream analysis.
    # You can override the output path by setting:
    #   export OUTPUT_CSV="/path/to/posthog_prs.csv"
    output_csv = os.environ.get("OUTPUT_CSV")
    if not output_csv:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_csv = f"posthog_prs_last90days_{ts}.csv"

    df.to_csv(output_csv, index=False)
    logger.info("Saved %s rows to CSV: %s", len(df), output_csv)

    # Compute and persist per-user aggregated features.
    user_features_df = compute_user_features(df, recent_days=30)

    features_csv = os.environ.get("OUTPUT_FEATURES_CSV")
    if not features_csv:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        features_csv = f"posthog_pr_user_features_last90days_{ts}.csv"

    user_features_df.to_csv(features_csv, index=False)
    logger.info("Saved per-user features to CSV: %s", features_csv)

    # Print a small sample for quick inspection.
    print(user_features_df.head(10).to_string(index=False))