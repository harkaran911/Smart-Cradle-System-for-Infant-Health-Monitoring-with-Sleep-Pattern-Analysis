#!/usr/bin/env python3
"""
daily_sleep_report.py
- Reads motion events from sleep_log.csv or smart_cradle.db
- Groups by day and generates a PDF per day with an embedded, stage-colored timeline
- Computes: total motion events, total motion minutes, estimated sleep (quiet gaps), window length, sleep efficiency
- Optional CLI flags for thresholds and date filtering
"""

import os
import io
import sys
import argparse
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from datetime import datetime, date
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# -------------------- Data Loaders --------------------

def load_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"] = pd.to_datetime(df["End"])
    return df.sort_values("Start").reset_index(drop=True)[["Start", "End"]]


def load_from_db(db_path: str, table_candidates=None) -> pd.DataFrame:
    if table_candidates is None:
        table_candidates = ["motion_log", "sleep_log", "events", "activity"]
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No CSV found and DB not present: {db_path}")

    with sqlite3.connect(db_path) as conn:
        for name in table_candidates:
            try:
                q = f'SELECT * FROM "{name}" ORDER BY Start;'
                df = pd.read_sql_query(q, conn)
                if df.empty:
                    continue
                # find likely columns
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

    raise RuntimeError("Could not find a usable table in smart_cradle.db. "
                       "Edit table_candidates or verify schema.")


def load_events(csv_path="sleep_log.csv", db_path="smart_cradle.db") -> pd.DataFrame:
    # Try CSV first; fallback to DB
    try:
        if os.path.exists(csv_path):
            return load_from_csv(csv_path)
    except Exception:
        pass
    return load_from_db(db_path)


# -------------------- Analytics Helpers --------------------

def add_gaps_and_stage(df: pd.DataFrame,
                       deep_min: float = 5.0,
                       light_min_low: float = 1.0,
                       light_min_high: float = 5.0) -> pd.DataFrame:
    """Adds NextStart, GapMin and Stage columns based on quiet-gap thresholds (minutes)."""
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


def sleep_summary(df_day: pd.DataFrame, quiet_min: float = 1.0) -> dict:
    """Compute daily metrics: events, motion_min, est sleep (quiet gaps), window, efficiency."""
    total_events = len(df_day)
    total_motion = (df_day["End"] - df_day["Start"]).sum()
    motion_min = total_motion.total_seconds()/60 if total_events else 0

    if total_events:
        span_min = (df_day["End"].max() - df_day["Start"].min()).total_seconds()/60
    else:
        span_min = 0

    gaps = (df_day["Start"].shift(-1) - df_day["End"]).dt.total_seconds()/60
    sleep_min = gaps[gaps >= quiet_min].sum() if gaps is not None else 0
    eff = (sleep_min/span_min)*100 if span_min else 0

    return {
        "events": total_events,
        "motion_min": motion_min,
        "sleep_min": sleep_min,
        "window_min": span_min,
        "sleep_eff_pct": eff
    }


def day_groups(df: pd.DataFrame, target_date=None):
    if target_date is not None:
        # filter to that date only
        mask = df["Start"].dt.date == target_date
        df = df[mask]
    return list(df.groupby(df["Start"].dt.date))


# -------------------- Plotting --------------------

def timeline_figure(df_staged: pd.DataFrame, quiet_min: float = 1.0):
    """Create a stage-colored timeline and shade quiet gaps >= quiet_min."""
    fig, ax = plt.subplots(figsize=(8.4, 2.2))

    color_map = {
        "Deep Sleep": "tab:blue",
        "Light Sleep": "tab:green",
        "Awake/Restless": "tab:red",
        "Unknown": "gray"
    }

    # draw motion segments
    for _, row in df_staged.iterrows():
        c = color_map.get(row["Stage"], "tab:gray")
        ax.plot([row["Start"], row["End"]], [1, 1], linewidth=8,
                color=c, solid_capstyle="butt")

    # shade quiet gaps
    for i in range(len(df_staged) - 1):
        gstart, gend = df_staged["End"].iloc[i], df_staged["Start"].iloc[i+1]
        gap = (gend - gstart).total_seconds() / 60.0
        if gap >= quiet_min:
            ax.axvspan(gstart, gend, facecolor="lightblue", alpha=0.25, ymin=0.2, ymax=0.8)

    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title("Infant Sleep/Motion Timeline (Colored by Stage)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# -------------------- PDF Writer --------------------

def write_daily_pdf(pdf_path: str, date_key, summary: dict, chart_png_bytes: bytes):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    W, H = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, H-50, "DAILY INFANT SLEEP REPORT")

    c.setFont("Helvetica", 12)
    c.drawString(50, H-90,  f"Date: {date_key}")
    c.drawString(50, H-115, f"Total Motion Events: {summary['events']}")
    c.drawString(50, H-140, f"Total Motion Duration: {summary['motion_min']:.2f} min")
    c.drawString(50, H-165, f"Estimated Sleep (quiet gaps ≥ 1m): {summary['sleep_min']:.2f} min")
    c.drawString(50, H-190, f"Monitoring Window: {summary['window_min']:.2f} min")
    c.drawString(50, H-215, f"Sleep Efficiency: {summary['sleep_eff_pct']:.1f}%")

    # Chart
    img = ImageReader(io.BytesIO(chart_png_bytes))
    c.drawImage(img, 50, 250, width=500, height=200,
                preserveAspectRatio=True, mask='auto')

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 60, "Generated by Smart Cradle System")
    c.showPage()
    c.save()


# -------------------- CLI & Main --------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate per-day infant sleep PDFs with stage-colored timelines."
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
                   help="Minimum quiet gap (min) considered as sleep when shading / summarizing (default: 1)")
    p.add_argument("--date", type=str, default=None,
                   help="Generate only for this date (YYYY-MM-DD)")
    p.add_argument("--out-dir", type=str, default=".",
                   help="Directory to write PDFs (default: current dir)")
    p.add_argument("--summary-csv", type=str, default=None,
                   help="If set, also write a daily_summary.csv with metrics")
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    try:
        df = load_events(args.csv, args.db)
    except Exception as e:
        print(f"[!] Failed to load data: {e}")
        sys.exit(1)

    if df.empty:
        print("[i] No events to report.")
        sys.exit(0)

    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("[!] --date must be in YYYY-MM-DD format.")
            sys.exit(1)

    groups = day_groups(df, target_date=target_date)
    if not groups:
        print("[i] No matching days found for reporting.")
        sys.exit(0)

    summaries = []

    for dkey, group in groups:
        staged = add_gaps_and_stage(
            group,
            deep_min=args.deep_min,
            light_min_low=args.light_min_low,
            light_min_high=args.light_min_high
        )

        # Make chart image
        fig = timeline_figure(staged, quiet_min=args.quiet_min)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        chart_png = buf.read()
        plt.close(fig)

        # Summaries
        summ = sleep_summary(staged, quiet_min=args.quiet_min)
        summaries.append({
            "date": dkey,
            **summ
        })

        # PDF path
        os.makedirs(args.out_dir, exist_ok=True)
        pdf_name = f"sleep_report_{dkey}.pdf"
        pdf_path = os.path.join(args.out_dir, pdf_name)
        write_daily_pdf(pdf_path, dkey, summ, chart_png)
        print(f"✅ Generated: {pdf_path}")

    if args.summary_csv:
        out_csv = os.path.join(args.out_dir, args.summary_csv)
        pd.DataFrame(summaries).to_csv(out_csv, index=False)
        print(f"📝 Wrote daily summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
