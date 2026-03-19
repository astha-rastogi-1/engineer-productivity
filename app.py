import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="PR Impact Score Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


WEIGHTS: Dict[str, float] = {
    # Must match the defaults used in `score_users_from_op.py`
    "w_merge_rate": 0.25,
    "w_total_prs_merged": 0.20,
    "w_time_to_merge_inv": 0.20,
    "w_product_impact_ratio": 0.15,
    "w_activity_score": 0.10,
    "w_days_active": 0.05,
    "w_recent_prs": 0.05,
    "w_stability": 0.05,
}

# Normalized metric columns in `scored_op.csv` that contribute to `impact_score`.
CONTRIBUTING_METRICS = [
    ("merge_rate_norm", "w_merge_rate", WEIGHTS["w_merge_rate"]),
    ("total_prs_merged_norm", "w_total_prs_merged", WEIGHTS["w_total_prs_merged"]),
    ("median_time_to_merge_inv_norm", "w_time_to_merge_inv", WEIGHTS["w_time_to_merge_inv"]),
    ("product_impact_ratio_norm", "w_product_impact_ratio", WEIGHTS["w_product_impact_ratio"]),
    ("activity_score_norm", "w_activity_score", WEIGHTS["w_activity_score"]),
    ("days_active_norm", "w_days_active", WEIGHTS["w_days_active"]),
    ("recent_prs_norm", "w_recent_prs", WEIGHTS["w_recent_prs"]),
    ("stability_norm", "w_stability", WEIGHTS["w_stability"]),
]

METRIC_DESCRIPTIONS: Dict[str, str] = {
    "merge_rate_norm": "Merge rate = merged PRs ÷ created PRs. Higher means their PRs get merged more reliably.",
    "total_prs_merged_norm": "Total merged PRs. Higher means more merged output (throughput).",
    "median_time_to_merge_inv_norm": "Inverse of the median time-to-merge. Higher means faster PR turnaround.",
    "product_impact_ratio_norm": 'Product impact ratio = merged PRs where `base_ref` == "main" ÷ total PRs. Higher means more impact on the mainline.',
    "activity_score_norm": "Activity score = total merged PRs × (1 / median time-to-merge). Higher means frequent and fast merges together.",
    "days_active_norm": "Unique days with PR activity. Higher means more consistent day-to-day contributions.",
    "recent_prs_norm": "PRs created in the most recent window (30 days). Higher means more current activity.",
    "stability_norm": "Stability = variance of time-to-merge (lower generally means more consistent turnaround). Interpreted relatively after normalization.",
}


def _repo_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@st.cache_data(show_spinner=False)
def load_scored_op(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find scored CSV at: {csv_path}. "
            f"Make sure `scored_op.csv` is present in the app directory."
        )
    df = pd.read_csv(csv_path)
    return df


def _safe_idxmax(s: pd.Series) -> int:
    # Returns index of maximum ignoring NaNs.
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.notna().sum() == 0:
        return -1
    return int(s2.idxmax())


def main() -> None:
    csv_path = os.environ.get("SCORED_OP_CSV", os.path.join(_repo_dir(), "scored_op.csv"))

    st.title("Top Engineers by PR Impact Score")

    try:
        df = load_scored_op(csv_path)
    except Exception as e:
        st.error(str(e))
        return

    required = ["author", "impact_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"scored_op.csv is missing required columns: {missing}")
        return

    # Keep numeric columns numeric.
    for col in df.columns:
        if col == "author":
            continue
        if col in ("created_at", "merged_at"):
            continue
        # Best-effort coercion; string columns stay as-is.
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Exclude bot accounts from the dashboard.
    if "author" in df.columns:
        bot_mask = df["author"].astype(str).str.contains(r"\[bot\]", case=False, na=False)
        df = df.loc[~bot_mask].copy()

    # Top 5 engineers by impact_score.
    df = df.sort_values("impact_score", ascending=False).reset_index(drop=True)
    top5 = df.head(5).copy()

    def snake_to_camel(s: str) -> str:
        base = s
        if base.endswith("_norm"):
            base = base[: -len("_norm")]
        parts = [p for p in base.split("_") if p]
        if not parts:
            return base
        return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])

    metric_to_weight = {col: weight for (col, _, weight) in CONTRIBUTING_METRICS}
    metric_cols = [col for (col, _, _) in CONTRIBUTING_METRICS if col in top5.columns]

    # For the contribution table we show non-normalized values (no inverses),
    # while keeping charts/metric descriptions based on the normalized metrics.
    TABLE_VALUE_COL: Dict[str, str] = {
        "merge_rate_norm": "merge_rate",
        "total_prs_merged_norm": "total_prs_merged",
        "median_time_to_merge_inv_norm": "median_time_to_merge",  # explicitly not the inverse
        "product_impact_ratio_norm": "product_impact_ratio",
        "activity_score_norm": "activity_score",
        "days_active_norm": "days_active",
        "recent_prs_norm": "recent_prs",
        "stability_norm": "stability",
    }

    # Header labels: camelCase without `_norm` and without the `Inv` part.
    TABLE_HEADER_BASE_COL: Dict[str, str] = {
        "median_time_to_merge_inv_norm": "median_time_to_merge",
    }

    def metric_header(metric_col: str) -> str:
        w = metric_to_weight.get(metric_col, 0.0)
        base = TABLE_HEADER_BASE_COL.get(metric_col, metric_col)
        return f"{snake_to_camel(base)} (w={w:.2f})"

    # Sidebar controls.
    st.sidebar.header("Controls")
    selected_metric = st.sidebar.selectbox(
        "Metric (normalized)",
        options=metric_cols,
        format_func=lambda m: snake_to_camel(m),
        index=0 if metric_cols else None,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Weights are hardcoded to match `score_users_from_op.py` defaults. "
        "If you change weights there, update them in this file too."
    )

    # Metric descriptions for all weighted metrics (for transparency).
    st.sidebar.subheader("Weighted Metric Descriptions")
    for metric_col, _weight_key, weight in CONTRIBUTING_METRICS:
        if metric_col not in metric_cols:
            continue
        label = snake_to_camel(metric_col)
        desc = METRIC_DESCRIPTIONS.get(metric_col, "—")
        st.sidebar.markdown(f"**{label}** (w={weight:.2f}): {desc}")

    # Contribution table (values only; weights in column headers).
    st.subheader("Top 5 Engineers")

    table_df = top5[["author", "impact_score"]].rename(
        columns={"author": "engineerName", "impact_score": "impactScore"}
    )

    for metric_col in metric_cols:
        value_col = TABLE_VALUE_COL.get(metric_col)
        if not value_col or value_col not in top5.columns:
            table_df[metric_col] = np.nan
            continue

        table_df[metric_col] = top5[value_col]

    table_df = table_df.rename(columns={m: metric_header(m) for m in metric_cols})

    # Round numeric columns for cleaner UI; format time-to-merge with units.
    for col in table_df.columns:
        if col == "engineerName":
            continue
        if col.startswith("medianTimeToMerge") or col.startswith("medianTimeToMerge ("):
            # Convert to "X.XXXX days" string for clarity.
            def _fmt_days(v):
                v_num = pd.to_numeric(v, errors="coerce")
                if pd.isna(v_num):
                    return "NA"
                return f"{float(v_num):.4f} days"

            table_df[col] = table_df[col].map(_fmt_days)
        else:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").round(4)
    st.dataframe(table_df, use_container_width=True)

    # Bar chart: separate bars for each of top-5 engineers (single metric).
    st.subheader("Normalized Metric Comparison")
    if selected_metric and selected_metric in top5.columns:
        if selected_metric in METRIC_DESCRIPTIONS:
            st.caption(METRIC_DESCRIPTIONS[selected_metric])
        metric_series = top5.set_index("author")[selected_metric]
        metric_series = pd.to_numeric(metric_series, errors="coerce")
        st.bar_chart(metric_series, use_container_width=True)

    # Insights (rule-based, no LLM).
    st.subheader("Insights")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    def _format_days(x: float) -> str:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.2f} days"

    def generate_insights(full_df: pd.DataFrame, top_row: pd.Series) -> List[str]:
        insights: List[str] = []
        if full_df.empty:
            return insights

        # 1) Top impact score + drivers (top 2 contribution terms).
        top_engineer = str(top_row["author"])
        impact = float(top_row["impact_score"]) if not pd.isna(top_row["impact_score"]) else None

        contribution_terms = []
        contribution_phrase = {
            "merge_rate_norm": "strong merge rate",
            "total_prs_merged_norm": "many merged PRs",
            "median_time_to_merge_inv_norm": "fast PR turnaround",
            "product_impact_ratio_norm": "high product impact",
            "activity_score_norm": "high overall activity",
            "days_active_norm": "activity across many days",
            "recent_prs_norm": "strong recent PR activity",
            "stability_norm": "turnaround variance (stability)",
        }
        for metric_col, _, weight in CONTRIBUTING_METRICS:
            if metric_col not in top_row.index:
                continue
            val = top_row[metric_col]
            val_num = pd.to_numeric(val, errors="coerce")
            if pd.isna(val_num):
                val_num = 0.0
            contribution_terms.append((float(weight) * float(val_num), metric_col))
        contribution_terms.sort(reverse=True, key=lambda x: x[0])
        top_drivers = [m for _, m in contribution_terms[:2] if m in contribution_phrase]
        if impact is not None and top_drivers:
            drivers_text = " and ".join(contribution_phrase[m] for m in top_drivers)
            insights.append(
                f"{top_engineer} leads with impact score {impact:.3f}, driven by {drivers_text}."
            )

        # 2) Fastest (max median_time_to_merge_inv_norm).
        if "median_time_to_merge_inv_norm" in full_df.columns:
            inv_series = pd.to_numeric(full_df["median_time_to_merge_inv_norm"], errors="coerce")
            if inv_series.notna().any():
                fastest_idx = int(inv_series.idxmax())
                fastest = full_df.loc[fastest_idx]
                fastest_name = str(fastest["author"])
                median_days = _format_days(fastest.get("median_time_to_merge"))
                insights.append(f"{fastest_name} has the fastest PR turnaround (median merge time: {median_days}).")

        # 3) Most consistent (min stability).
        if "stability" in full_df.columns:
            stab_series = pd.to_numeric(full_df["stability"], errors="coerce")
            stab_series = stab_series.dropna()
            if not stab_series.empty:
                consistent_idx = int(stab_series.idxmin())
                consistent = full_df.loc[consistent_idx]
                consistent_name = str(consistent["author"])
                insights.append(
                    f"{consistent_name} is the most consistent (lowest time-to-merge variance: {float(consistent['stability']):.3f})."
                )

        # 4) Most active (max days_active).
        if "days_active" in full_df.columns:
            active_series = pd.to_numeric(full_df["days_active"], errors="coerce")
            active_series = active_series.dropna()
            if not active_series.empty:
                active_idx = int(active_series.idxmax())
                active = full_df.loc[active_idx]
                insights.append(f"{str(active['author'])} is the most active across days (days_active={int(active['days_active'])}).")

        # 5) Highest product impact (max product_impact_ratio).
        if "product_impact_ratio" in full_df.columns:
            pir_series = pd.to_numeric(full_df["product_impact_ratio"], errors="coerce")
            pir_series = pir_series.dropna()
            if not pir_series.empty:
                pir_idx = int(pir_series.idxmax())
                pir_row = full_df.loc[pir_idx]
                insights.append(
                    f"{str(pir_row['author'])} has the highest product impact (product_impact_ratio={float(pir_row['product_impact_ratio']):.3f})."
                )

        # Ensure we return 3-5 insights.
        return insights[:5]

    top_row = top5.iloc[0] if not top5.empty else df.iloc[0]
    insights = generate_insights(df, top_row)

    for s in insights:
        st.write(f"- {s}")

    st.caption(f"Computed from `scored_op.csv` ({now_utc} UTC).")


if __name__ == "__main__":
    main()

