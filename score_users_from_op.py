#!/usr/bin/env python3
"""
Read `op.csv` and compute activity + normalized metrics, then a weighted impact score.

Outputs:
  - scored CSV (with *_norm columns + impact_score)
  - prints top 5 users by impact_score
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("score_users_from_op")


# Editable weights for impact score (sum/scale doesn't matter).
# Tip: if you want "lower stability is better", set `w_stability` negative.
WEIGHTS: Dict[str, float] = {
    "w_merge_rate": 0.25,
    "w_total_prs_merged": 0.20,
    "w_time_to_merge_inv": 0.20,  # median_time_to_merge inverse (higher is better)
    "w_product_impact_ratio": 0.15,
    "w_activity_score": 0.10,
    "w_days_active": 0.05,
    "w_recent_prs": 0.05,
    "w_stability": 0.05,
}


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Min-max scale to [0, 1]. Handles all-NaN and constant vectors.
    Missing values become 0 so they don't contribute to impact_score.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    min_val = float(s.min(skipna=True))
    max_val = float(s.max(skipna=True))
    if max_val == min_val:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    norm = (s - min_val) / (max_val - min_val)
    norm = norm.fillna(0.0)
    return norm


def safe_inverse_days(series: pd.Series) -> pd.Series:
    """
    Compute 1/median_time_to_merge with safe division.
    Returns NaN for 0, negative, or missing values.
    """
    s = pd.to_numeric(series, errors="coerce")
    return np.where((s.isna()) | (s <= 0), np.nan, 1.0 / s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute activity + impact score from op.csv")
    parser.add_argument(
        "--input-csv",
        default=os.environ.get("INPUT_CSV", "op.csv"),
        help="Path to op.csv (default: op.csv or $INPUT_CSV).",
    )
    parser.add_argument(
        "--output-csv",
        default="scored_op.csv",
        help="Path to write scored CSV (default: timestamped in CWD).",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)

    input_csv = args.input_csv
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    logger.info("Loading: %s", input_csv)
    df = pd.read_csv(input_csv)
    logger.info("Loaded %s rows x %s cols", len(df), df.shape[1])

    # Required columns from op.csv
    required = [
        "author",
        "total_prs_merged",
        "median_time_to_merge",
        "merge_rate",
        "product_impact_ratio",
        "days_active",
        "recent_prs",
        "stability",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Activity score: total_prs_merged * (1 / median_time_to_merge)
    median_inv = safe_inverse_days(df["median_time_to_merge"])
    df["median_time_to_merge_inv"] = median_inv
    df["activity_score"] = pd.to_numeric(df["total_prs_merged"], errors="coerce") * pd.to_numeric(
        df["median_time_to_merge_inv"], errors="coerce"
    )
    # If median_time_to_merge missing/0 => activity_score will be NaN; normalize will treat as 0.

    metrics_to_normalize = {
        "merge_rate": df["merge_rate"],
        "total_prs_merged": df["total_prs_merged"],
        "median_time_to_merge_inv": df["median_time_to_merge_inv"],
        "product_impact_ratio": df["product_impact_ratio"],
        "activity_score": df["activity_score"],
        "days_active": df["days_active"],
        "recent_prs": df["recent_prs"],
        "stability": df["stability"],
    }

    # Create *_norm columns
    for metric_name, series in metrics_to_normalize.items():
        df[f"{metric_name}_norm"] = min_max_normalize(series)

    # Weighted impact score using normalized metrics.
    df["impact_score"] = (
        WEIGHTS["w_merge_rate"] * df["merge_rate_norm"]
        + WEIGHTS["w_total_prs_merged"] * df["total_prs_merged_norm"]
        + WEIGHTS["w_time_to_merge_inv"] * df["median_time_to_merge_inv_norm"]
        + WEIGHTS["w_product_impact_ratio"] * df["product_impact_ratio_norm"]
        + WEIGHTS["w_activity_score"] * df["activity_score_norm"]
        + WEIGHTS["w_days_active"] * df["days_active_norm"]
        + WEIGHTS["w_recent_prs"] * df["recent_prs_norm"]
        + WEIGHTS["w_stability"] * df["stability_norm"]
    )

    top5 = df.sort_values("impact_score", ascending=False).head(5)
    cols = [
        "author",
        "impact_score",
        "merge_rate",
        "total_prs_merged",
        "median_time_to_merge",
        "median_time_to_merge_inv",
        "product_impact_ratio",
        "activity_score",
        "days_active",
        "recent_prs",
        "stability",
    ]
    logger.info("Top 5 users by impact_score:")
    print(top5[cols].to_string(index=False))

    # Save scored dataframe
    output_csv = args.output_csv
    if not output_csv:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_csv = f"op_scored_{ts}.csv"

    df.to_csv(output_csv, index=False)
    logger.info("Saved scored output to: %s", output_csv)


if __name__ == "__main__":
    main()

