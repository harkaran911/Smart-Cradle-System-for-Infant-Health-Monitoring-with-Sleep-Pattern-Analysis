"""
sleep_stage_estimation.py
- Reads infant motion intervals from sleep_log.csv or smart_cradle.db
- Classifies sleep stage from quiet gaps between motion events
- Saves per-day stage CSVs and stage-colored timeline PNGs (mini hypnograms)
- Optional master summary CSV

Stages (default thresholds; all tunable via CLI):
  gap > deep_min (5m)         -> Deep Sleep
  light_min_low..light_min_high (1..5m) -> Light Sleep
  gap < light_min_low (1m)    -> Awake/Restless
  last row (no next gap)      -> Unknown
"""

import os
import io
import sys
import argparse
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
from datetime import datetime, date


# -------------------- Data Loading --------------------

def load_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"] = pd.to_datetime(df["End"])
    return df.sort_values("Start").reset_index(drop=True)[["Start", "End"]]


def load_from_db(db_path: str, table_candidates=None) -> pd.DataFrame:
    if table_candidates is None:
        table_candidates = ["motion_log", "sleep_log", "events", "activity"]

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        for name in table_candidates:
            try:
                q = f'SELECT * FROM "{name}" ORDER BY Start;'
                df = pd.read_sql_query(q, conn)
                if df.empty:
                    continue
                cand_start = [c for c in df.columns if c.lower().startswith("start")]
                cand_end   = [c for c in df.columns if c.lower().startswith("end")]
                if not cand_start or not cand_end:
                    continue
                df.rename(columns={cand_start[0]: "Start", cand_end[0]: "End"}, inplace=True)
                df["Start"] = pd.to_datetime(df["Start"])
                df["End"]   = pd.to_datetime(df["End"])
                return df.sort_values("Start").reset_index(drop=True)[["Start", "End"]]
            except Exception:
                continue

    raise RuntimeError("Could not find a usable table in the DB. "
                       "Edit table_candidates or verify schema.")


def load_events(csv_path="sleep_log.csv", db_path="smart_cradle.db") -> pd.DataFrame:
    try:
        if os.path.exists(csv_path):
            return load_from_csv(csv_path)
    except Exception:
        pass
    return load_from_db(db_path)


# -------------------- Stage Classification --------------------

def add_gaps_and_stage(df: pd.DataFrame,
                       deep_min: float = 5.0,
                       light_min_low: float = 1.0,
                       light_min_high: float = 5.0) -> pd.DataFrame:
    """
    Adds columns:
      - NextStart: timestamp of next event's Start
      - GapMin: minutes of quiet time between End[i] and Start[i+1]
      - Stage: classification from GapMin
    """
    df = df.copy()
    df["NextStart"] = df["Start"].shift(-1)
    df["GapMin"] = (df["NextStart"] - df["End"]).dt.total_seconds() / 60.0

    def classify(g):
        if pd.isna(g):
            return "Unknown"
        if g > deep_min:
            return "Deep Sleep"
        if light_min_low <= g <= light_min_high:
            return "Light Sleep"
        return "Awake/Restless"

    df["Stage"] = df["GapMin"].apply(classify)
    return df


def day_groups(df: pd.DataFrame, only_date=None):
    if only_date is not None:
        mask = df["Start"].dt.date == only_date
        df = df[mask]
    return list(df.groupby(df["Start"].dt.date))


# -------------------- Plotting --------------------

def stage_timeline_figure(df_staged: pd.DataFrame,
                          quiet_min: float = 1.0,
                          title_prefix: str = "Sleep Stages") -> matplotlib.figure.Figure:
    """
    Build a color-coded stage timeline:
      - Red = Awake/Restless, Green = Light, Blue = Deep, Gray = Unknown
      - Shaded light-blue spans for quiet gaps >= quiet_min
    """
    fig, ax = plt.subplots(figsize=(8.4, 2.2))

    color_map = {
        "Deep Sleep": "tab:blue",
        "Light Sleep": "tab:green",
        "Awake/Restless": "tab:red",
        "Unknown": "gray"
    }

    # motion segments colored by Stage
    for _, row in df_staged.iterrows():
        c = color_map.get(row["Stage"], "tab:gray")
        ax.plot([row["Start"], row["End"]], [1, 1],
                linewidth=8, color=c, solid_capstyle="butt")

    # quiet gaps shading (likely sleep if >= quiet_min)
    for i in range(len(df_staged) - 1):
        gstart, gend = df_staged["End"].iloc[i], df_staged["Start"].iloc[i+1]
        gap = (gend - gstart).total_seconds() / 60.0
        if gap >= quiet_min:
            ax.axvspan(gstart, gend, facecolor="lightblue", alpha=0.25, ymin=0.2, ymax=0.8)

    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title(f"{title_prefix} — Stage-Colored Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# -------------------- Summaries --------------------

def stage_counts(df_staged: pd.DataFrame) -> dict:
    counts = df_staged["Stage"].value_counts(dropna=False).to_dict()
    for key in ["Deep Sleep", "Light Sleep", "Awake/Restless", "Unknown"]:
        counts.setdefault(key, 0)
    return counts


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate sleep stages from quiet gaps and export per-day CSVs + stage timeline PNGs."
    )
    p.add_argument("--csv", default="sleep_log.csv",
                   help="Path to CSV (default: sleep_log.csv)")
    p.add_argument("--db", default="smart_cradle.db",
                   help="Path to SQLite DB (fallback if CSV missing)")
    p.add_argument("--deep-min", type=float, default=5.0,
                   help="Gap (min) above which counts as Deep Sleep (default: 5)")
    p.add_argument("--light-min-low", type=float, default=1.0,
                   help="Lower bound (min) for Light Sleep gap (default: 1)")
    p.add_argument("--light-min-high", type=float, default=5.0,
                   help="Upper bound (min) for Light Sleep gap (default: 5)")
    p.add_argument("--quiet-min", type=float, default=1.0,
                   help="Minimum quiet gap (min) shaded as sleep (default: 1)")
    p.add_argument("--date", type=str, default=None,
                   help="Only process this date (YYYY-MM-DD)")
    p.add_argument("--out-dir", type=str, default=".",
                   help="Directory to write outputs (default: current dir)")
    p.add_argument("--summary-csv", type=str, default=None,
                   help="If set, also write a per-day summary CSV (e.g., stage counts)")
    return p.parse_args()


# -------------------- Main --------------------

def main():
    args = parse_args()

    # Load events
    try:
        df = load_events(args.csv, args.db)
    except Exception as e:
        print(f"[!] Failed to load data: {e}")
        sys.exit(1)

    if df.empty:
        print("[i] No events found.")
        sys.exit(0)

    only_date = None
    if args.date:
        try:
            only_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("[!] --date must be YYYY-MM-DD.")
            sys.exit(1)

    groups = day_groups(df, only_date=only_date)
    if not groups:
        print("[i] No matching days to process.")
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)
    master_summary = []

    for dkey, g in groups:
        staged = add_gaps_and_stage(
            g,
            deep_min=args.deep_min,
            light_min_low=args.light_min_low,
            light_min_high=args.light_min_high
        )

        # ---- Save per-day staged CSV ----
        csv_name = f"sleep_stages_{dkey}.csv"
        csv_path = os.path.join(args.out_dir, csv_name)
        staged_out = staged[["Start", "End", "GapMin", "Stage"]].copy()
        staged_out.to_csv(csv_path, index=False)
        print(f"📝 Wrote: {csv_path}")

        # ---- Plot & save per-day stage timeline ----
        fig = stage_timeline_figure(staged, quiet_min=args.quiet_min,
                                    title_prefix=f"Sleep Stages — {dkey}")
        png_name = f"sleep_stages_{dkey}.png"
        png_path = os.path.join(args.out_dir, png_name)
        fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"🖼  Saved: {png_path}")

        # ---- Collect per-day summary (stage counts, motion minutes, etc.) ----
        counts = stage_counts(staged)
        total_motion = (g["End"] - g["Start"]).sum().total_seconds() / 60.0
        master_summary.append({
            "date": dkey,
            "Deep Sleep events": counts["Deep Sleep"],
            "Light Sleep events": counts["Light Sleep"],
            "Awake/Restless events": counts["Awake/Restless"],
            "Unknown events": counts["Unknown"],
            "Total motion minutes": round(total_motion, 2)
        })

    if args.summary_csv and master_summary:
        out_csv = os.path.join(args.out_dir, args.summary_csv)
        pd.DataFrame(master_summary).to_csv(out_csv, index=False)
        print(f"📊 Wrote summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
