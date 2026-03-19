#!/usr/bin/env python3
"""
Compute per-user PR metrics from the exported GitHub PR CSV.

Input: CSV produced by `test.py` (default name pattern: posthog_prs_last90days_*.csv)
Output: per-user aggregated CSV with:
  - total_prs_created
  - total_prs_merged
  - merge_rate = merged / created
  - median_time_to_merge
  - days_active (unique days with PR activity)
  - recent_prs (last 30 days)
  - product_impact_ratio = PRs merged into base_ref == "main" / total PRs created
  - stability = variance of time_to_merge
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("compute_user_pr_features")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def compute_user_features(
    df: pd.DataFrame,
    *,
    recent_days: int = 30,
) -> pd.DataFrame:
    # Expected input schema (from our PR exporter):
    # - author_login
    # - created_at (ISO timestamps)
    # - merged_at (ISO timestamps or empty)
    # - base_ref (branch name for PR base)
    required_cols = ["author_login", "created_at", "merged_at"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Parse timestamps (UTC); invalid values become NaT.
    df = df.copy()

    # Exclude bots (engineers reported by GitHub but marked as bots).
    if "author_login" in df.columns:
        bot_mask = df["author_login"].astype(str).str.contains(r"\[bot\]", case=False, na=False)
        df = df.loc[~bot_mask].copy()

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["merged_at"] = pd.to_datetime(df["merged_at"], utc=True, errors="coerce")

    # Time-to-merge (days) only for merged PRs.
    df["time_to_merge_days"] = (df["merged_at"] - df["created_at"]).dt.total_seconds() / 86400.0

    # Activity time window.
    now_utc = datetime.now(timezone.utc)
    recent_cutoff = now_utc - timedelta(days=recent_days)
    df["is_recent"] = df["created_at"] >= recent_cutoff
    df["is_merged"] = df["merged_at"].notna()

    # Base branch impact (from CSV export column `base_ref`).
    base_ref_present = "base_ref" in df.columns
    if base_ref_present:
        df["is_main_base"] = df["base_ref"] == "master"
    else:
        df["is_main_base"] = False

    grouped = df.groupby("author_login", dropna=False)

    def _days_active(s: pd.Series) -> int:
        # Unique UTC dates where they created at least one PR.
        # Normalize to midnight and drop NaT.
        dates = s.dropna().dt.normalize()
        return int(dates.nunique())

    def _median_time_to_merge(s: pd.Series) -> Optional[float]:
        vals = s.dropna().to_numpy()
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _variance_time_to_merge(s: pd.Series) -> Optional[float]:
        vals = s.dropna().to_numpy()
        if vals.size == 0:
            return None
        # Use population variance (ddof=0) so a single PR gives variance=0.
        return float(np.var(vals, ddof=0))

    agg = grouped.agg(
        total_prs_created=("created_at", "count"),
        total_prs_merged=("is_merged", "sum"),
        days_active=("created_at", _days_active),
        recent_prs=("is_recent", "sum"),
        median_time_to_merge=("time_to_merge_days", _median_time_to_merge),
        stability=("time_to_merge_days", _variance_time_to_merge),
        main_base_prs=("is_main_base", "sum"),
    ).reset_index()

    # Derived ratios.
    agg["merge_rate"] = agg.apply(
        lambda r: (r["total_prs_merged"] / r["total_prs_created"]) if r["total_prs_created"] else None,
        axis=1,
    )
    agg["product_impact_ratio"] = agg.apply(
        lambda r: (r["main_base_prs"] / r["total_prs_created"]) if r["total_prs_created"] else None,
        axis=1,
    )

    # Clean up internal helper column.
    agg = agg.rename(columns={"author_login": "author"})
    agg = agg.drop(columns=["main_base_prs"])

    # Optional logging about schema compatibility.
    if not base_ref_present:
        logger.warning(
            "Column `base_ref` not found in input CSV; `product_impact_ratio` will be null/0-derived."
        )

    # Reorder output columns (keeps a stable schema).
    out_cols = [
        "author",
        "total_prs_created",
        "total_prs_merged",
        "merge_rate",
        "median_time_to_merge",
        "days_active",
        "recent_prs",
        "product_impact_ratio",
        "stability",
    ]
    return agg[out_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-user PR metrics from a PR CSV export.")
    parser.add_argument(
        "--input-csv",
        default=os.environ.get("INPUT_CSV", "data.csv"),
        help="Path to the input PR CSV (default: data.csv or $INPUT_CSV).",
    )
    parser.add_argument(
        "--output-csv",
        default=os.environ.get("OUTPUT_CSV"),
        help="Path to write aggregated per-user CSV (default: timestamped in CWD or $OUTPUT_CSV).",
    )
    parser.add_argument("--recent-days", type=int, default=30, help="Window for `recent_prs` (default: 30).")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"), help="Logging level (default: INFO).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_csv = args.input_csv
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    logger.info("Loading input CSV: %s", input_csv)
    df = pd.read_csv(input_csv)
    logger.info("Loaded %s rows and %s columns", len(df), df.shape[1])

    out_df = compute_user_features(df, recent_days=args.recent_days)
    logger.info("Computed per-user features for %s users", len(out_df))

    output_csv = args.output_csv
    if not output_csv:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_csv = f"user_pr_features_{ts}.csv"

    out_df.to_csv(output_csv, index=False)
    logger.info("Saved aggregated per-user features to: %s", output_csv)

    # Print head for quick visibility.
    print(out_df.head().to_string(index=False))


if __name__ == "__main__":
    main()

