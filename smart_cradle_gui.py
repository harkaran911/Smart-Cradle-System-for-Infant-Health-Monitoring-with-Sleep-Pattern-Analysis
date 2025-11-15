import os
import io
import cv2
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import queue, threading, subprocess
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter

BG_COLOR = "#0A192F"        
PANEL_COLOR = "#112240"     
BTN_COLOR = "#1F6FEB"       
BTN_HOVER = "#58A6FF"       
TXT_COLOR = "#E6EDF3"       
TITLE_COLOR = "#64FFDA"     
DANGER_COLOR = "#ef4444"    

FONT_MAIN = ("Segoe UI", 12)
FONT_TITLE = ("Century Gothic", 18, "bold")

cap = None
running = False
first_frame = None
status_list = [0, 0]    
times = []              
df_session = None        
idle_frames = 0          
video_label = None      

btn_start_webcam = None
btn_start_file = None
btn_stop = None

status_bar = None

audio_running = False
audio_thread = None
audio_q = queue.Queue(maxsize=20)

cry_state = False           
cry_started_at = None
cry_events = []            
cry_conf_hist = []          

auto_soothe_var = None      
last_soothe_time = None     
SOOTHE_COOLDOWN_SEC = 60   

def make_glass_button(parent, text, command):
    btn = tk.Label(parent, text=text, bg=BTN_COLOR, fg="white",
                   font=("Segoe UI", 11, "bold"), padx=14, pady=8, cursor="hand2")
    btn.configure(relief="flat", bd=0, highlightthickness=0)
    btn.bind("<Button-1>", lambda e: command())
    btn.bind("<Enter>", lambda e: btn.config(bg=BTN_HOVER))
    btn.bind("<Leave>", lambda e: btn.config(bg=BTN_COLOR))
    return btn

def select_video_file():
    return filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )

def ensure_db():
    if not os.path.exists("smart_cradle.db"):
        conn = sqlite3.connect("smart_cradle.db")
        conn.close()

def save_log():
    """Persist current session's motion intervals to CSV and SQLite."""
    global times
    if not times:
        return

    rows = []
    for (start, end) in times:
        if start is not None and end is not None:
            rows.append({"Start": start, "End": end})

    if not rows:
        return

    df = pd.DataFrame(rows)
    csv_file = "sleep_log.csv"
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_file, index=False)

    ensure_db()
    with sqlite3.connect("smart_cradle.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS motion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Start TEXT,
                End TEXT
            );
        """)
        df.to_sql("motion_log", conn, if_exists="append", index=False)

def save_cry_events(events):
    """
    events: list of (start_dt, end_dt, avg_conf)
    """
    if not events:
        return
    rows = [{"Start": s, "End": e, "Confidence": float(c)} for (s, e, c) in events]
    df = pd.DataFrame(rows)

    csv_file = "cry_log.csv"
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_file, index=False)

    ensure_db()
    with sqlite3.connect("smart_cradle.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cry_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Start TEXT,
                End TEXT,
                Confidence REAL
            );
        """)
        df.to_sql("cry_log", conn, if_exists="append", index=False)

def load_events(prefer="csv"):
    try:
        df = pd.read_csv("sleep_log.csv")
        df["Start"] = pd.to_datetime(df["Start"])
        df["End"] = pd.to_datetime(df["End"])
        df = df.sort_values("Start").reset_index(drop=True)
        return df[["Start", "End"]]
    except Exception:
        pass

    ensure_db()
    TABLE_CANDIDATES = ["motion_log", "sleep_log", "events", "activity"]
    with sqlite3.connect("smart_cradle.db") as conn:
        for name in TABLE_CANDIDATES:
            try:
                q = f'SELECT * FROM "{name}" ORDER BY Start;'
                df = pd.read_sql_query(q, conn)
                cand_start = [c for c in df.columns if c.lower().startswith("start")]
                cand_end   = [c for c in df.columns if c.lower().startswith("end")]
                if not cand_start or not cand_end:
                    continue
                df.rename(columns={cand_start[0]: "Start", cand_end[0]: "End"}, inplace=True)
                df["Start"] = pd.to_datetime(df["Start"])
                df["End"]   = pd.to_datetime(df["End"])
                return df[["Start", "End"]].sort_values("Start").reset_index(drop=True)
            except Exception:
                continue

    raise RuntimeError("No sleep_log.csv or usable table in smart_cradle.db found.")

def add_gaps_and_stage(df, deep_min=5, light_min_low=1, light_min_high=5):
    """Compute gap→stage and return a copy with columns: Start, End, NextStart, GapMin, Stage."""
    df = df.copy()
    df["NextStart"] = df["Start"].shift(-1)
    df["GapMin"] = (df["NextStart"] - df["End"]).dt.total_seconds() / 60.0

    def classify(g):
        if pd.isna(g): return "Unknown"
        if g > deep_min: return "Deep Sleep"
        if light_min_low <= g <= light_min_high: return "Light Sleep"
        return "Awake/Restless"

    df["Stage"] = df["GapMin"].apply(classify)
    return df

def day_groups(df):
    """List of (date, group_df)."""
    return list(df.groupby(df["Start"].dt.date))

def sleep_summary(df_day):
    """Compute simple daily metrics."""
    total_events = len(df_day)
    total_motion = (df_day["End"] - df_day["Start"]).sum()
    motion_min = total_motion.total_seconds()/60 if total_events else 0

    if total_events:
        span_min = (df_day["End"].max() - df_day["Start"].min()).total_seconds()/60
    else:
        span_min = 0

    gaps = (df_day["Start"].shift(-1) - df_day["End"]).dt.total_seconds()/60
    sleep_min = gaps[gaps >= 1].sum() if gaps is not None else 0
    eff = (sleep_min/span_min)*100 if span_min else 0
    return {
        "events": total_events,
        "motion_min": motion_min,
        "sleep_min": sleep_min,
        "window_min": span_min,
        "sleep_eff_pct": eff
    }

def timeline_figure(df, color_by_stage=False):
    """Create a motion timeline figure; optionally color by stage and shade quiet gaps."""
    fig, ax = plt.subplots(figsize=(8, 2))

    if color_by_stage and "Stage" in df.columns:
        color_map = {"Deep Sleep":"tab:blue", "Light Sleep":"tab:green",
                     "Awake/Restless":"tab:red", "Unknown":"gray"}
        for _, row in df.iterrows():
            c = color_map.get(row["Stage"], "tab:gray")
            ax.plot([row["Start"], row["End"]], [1, 1], linewidth=8, color=c, solid_capstyle="butt")
    else:
        for _, row in df.iterrows():
            ax.plot([row["Start"], row["End"]], [1, 1], linewidth=8, color="tab:blue", solid_capstyle="butt")

    for i in range(len(df)-1):
        gstart, gend = df["End"].iloc[i], df["Start"].iloc[i+1]
        gap = (gend - gstart).total_seconds()/60
        if gap >= 1:
            ax.axvspan(gstart, gend, facecolor="lightblue", alpha=0.25, ymin=0.2, ymax=0.8)

    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title("Infant Sleep/Motion Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig

def fig_to_tk(fig, target_label):
    """Render a matplotlib fig to a Tk label."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf)
    imgtk = ImageTk.PhotoImage(im)
    target_label.imgtk = imgtk
    target_label.configure(image=imgtk)
    plt.close(fig)

def alerts_box_insert(text):
    try:
        alerts_box.insert("end", text + "\n")
        alerts_box.see("end")
    except Exception:
        pass

def start_monitoring(source="webcam"):
    global running, cap, first_frame, status_list, times, df_session, idle_frames
    try:
        if cap is not None and hasattr(cap, "isOpened") and cap.isOpened():
            cap.release()
    except Exception:
        pass
    cap = None

    running = True
    first_frame = None
    status_list = [0, 0]
    times = []
    df_session = pd.DataFrame(columns=["Start", "End"])
    idle_frames = 0

    if source == "webcam":
        cap_local = cv2.VideoCapture(0)
    else:
        video_path = select_video_file()
        if not video_path:
            status_bar.config(text="No file selected.")
            running = False
            return
        cap_local = cv2.VideoCapture(video_path)

    if (cap_local is None) or (not cap_local.isOpened()):
        running = False
        status_bar.config(text="Error: Could not open video source.")
        messagebox.showerror("Video Source", "Could not open video source.")
        return

    btn_start_webcam.config(state="disabled")
    btn_start_file.config(state="disabled")
    btn_stop.config(state="normal")

    status_bar.config(text="Monitoring Started...")
    globals()["cap"] = cap_local
    update_frame()

def stop_monitoring():
    global running, cap, status_list, times

    if not running and cap is None:
        return

    running = False
    try:
        if cap is not None and hasattr(cap, "isOpened") and cap.isOpened():
            cap.release()
    except Exception:
        pass
    cap = None

    if status_list[-1] == 1 and times:
        if times[-1][1] is None:
            times[-1] = (times[-1][0], datetime.now())

    save_log()
    status_bar.config(text="Monitoring Stopped")

    btn_start_webcam.config(state="normal")
    btn_start_file.config(state="normal")
    btn_stop.config(state="disabled")

def update_frame():
    """Main loop: read frame, detect motion, draw UI, schedule next frame."""
    global first_frame, status_list, times, running, idle_frames

    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        stop_monitoring()
        return

    target_w = 800
    h, w = frame.shape[:2]
    scale = target_w / float(w)
    new_w = target_w
    new_h = int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    status = 0  
    if first_frame is None:
        first_frame = gray
    else:
        delta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 7000:  
                continue
            status = 1
            (x, y, w2, h2) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w2, y+h2), (0, 255, 0), 2)

    status_list.append(status)
    status_list = status_list[-2:]

    now = datetime.now()
    if status_list[-2] == 0 and status_list[-1] == 1:
        times.append((now, None))
        status_bar.config(text="Motion detected…")
    if status_list[-2] == 1 and status_list[-1] == 0:
        if times and times[-1][1] is None:
            times[-1] = (times[-1][0], now)
        status_bar.config(text="Monitoring…")

    if status == 0:
        idle_frames += 1
        if idle_frames >= 500:
            first_frame = gray
            idle_frames = 0
    else:
        idle_frames = 0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(20, update_frame)  

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def frame_features(x, fs):
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    zc = float(((x[:-1] * x[1:]) < 0).mean())
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    mag_sum = np.sum(X) + 1e-12
    centroid = float(np.sum(freqs * X) / mag_sum)
    cumsum = np.cumsum(X)
    cutoff = 0.85 * cumsum[-1]
    rolloff_idx = np.searchsorted(cumsum, cutoff)
    rolloff = float(freqs[min(rolloff_idx, len(freqs)-1)])
    return rms, zc, centroid, rolloff

def cry_heuristic(rms, zc, centroid, rolloff,
                  rms_th=0.02, zc_min=0.02, zc_max=0.25,
                  cent_min=600, cent_max=3500, roll_min=1200):
    gates = [
        rms > rms_th,
        zc_min <= zc <= zc_max,
        cent_min <= centroid <= cent_max,
        rolloff >= roll_min
    ]
    score = sum(gates)
    conf = score / len(gates)
    return (score >= 3), conf

def audio_callback(indata, frames, time_info, status):
    mono = np.mean(indata, axis=1).astype(np.float32)
    try:
        audio_q.put_nowait(mono)
    except queue.Full:
        pass

def launch_soothe():
    try:
        if os.path.exists("cradle_rocker.py"):
            subprocess.Popen(["python", "cradle_rocker.py"])
            alerts_box_insert("🎵 Auto-Soothe: Rocker launched.")
        else:
            alerts_box_insert("⚠ Auto-Soothe requested but cradle_rocker.py not found.")
    except Exception as e:
        alerts_box_insert(f"⚠ Auto-Soothe failed: {e}")

def audio_worker(fs=16000, frame_sec=0.5, hop_sec=0.25,
                 lowcut=400, highcut=6000, consec_on=3, consec_off=4):
    global cry_state, cry_started_at, cry_conf_hist, cry_events, audio_running
    global last_soothe_time

    frame_len = int(frame_sec * fs)
    hop_len = int(hop_sec * fs)
    buffer = np.zeros(0, dtype=np.float32)
    on_count = 0
    off_count = 0

    while audio_running:
        try:
            chunk = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        buffer = np.concatenate([buffer, chunk])

        while len(buffer) >= frame_len:
            frame = buffer[:frame_len]
            buffer = buffer[hop_len:]

            y = bandpass_filter(frame, lowcut, highcut, fs)
            rms, zc, centroid, rolloff = frame_features(y, fs)
            is_cry, conf = cry_heuristic(rms, zc, centroid, rolloff)

            cry_conf_hist.append(conf)
            if len(cry_conf_hist) > 60:
                cry_conf_hist[:] = cry_conf_hist[-60:]

            if is_cry:
                on_count += 1; off_count = 0
            else:
                off_count += 1; on_count = 0

            now = datetime.now()
            if not cry_state and on_count >= consec_on:
                cry_state = True
                cry_started_at = now
                alerts_box_insert("🔊 Cry detected — episode started.")
                try: status_bar.config(text="🔊 Cry detected…")
                except: pass

                try:
                    if auto_soothe_var.get():
                        if (last_soothe_time is None) or ((now - last_soothe_time).total_seconds() >= SOOTHE_COOLDOWN_SEC):
                            last_soothe_time = now
                            launch_soothe()
                except Exception:
                    pass

            elif cry_state and off_count >= consec_off:
                cry_state = False
                if cry_started_at:
                    avg_conf = float(np.mean(cry_conf_hist[-(consec_on+consec_off):])) if cry_conf_hist else 0.0
                    cry_events.append((cry_started_at, now, avg_conf))
                    save_cry_events([(cry_started_at, now, avg_conf)])
                    alerts_box_insert(f"✅ Cry ended ({(now-cry_started_at).total_seconds():.1f}s).")
                    cry_started_at = None

def start_audio_monitor():
    global audio_running, audio_thread, cry_state, cry_started_at, cry_events, cry_conf_hist
    if audio_running:
        return
    audio_running = True
    cry_state = False
    cry_started_at = None
    cry_events = []
    cry_conf_hist = []

    sd.default.samplerate = 16000
    sd.default.channels = 1

    stream = sd.InputStream(callback=audio_callback)
    stream.start()

    def run():
        try:
            audio_worker(fs=sd.default.samplerate)
        finally:
            try: stream.stop()
            except: pass
            try: stream.close()
            except: pass

    audio_thread = threading.Thread(target=run, daemon=True)
    audio_thread.start()
    try: status_bar.config(text="Audio monitoring started.")
    except: pass

def stop_audio_monitor():
    global audio_running, audio_thread, cry_state, cry_started_at
    if not audio_running:
        return
    audio_running = False
    if audio_thread and audio_thread.is_alive():
        audio_thread.join(timeout=2.0)

    if cry_state and cry_started_at:
        now = datetime.now()
        avg_conf = float(np.mean(cry_conf_hist[-8:])) if cry_conf_hist else 0.0
        save_cry_events([(cry_started_at, now, avg_conf)])
        cry_events.append((cry_started_at, now, avg_conf))
        cry_state = False
        cry_started_at = None

    try: status_bar.config(text="Audio monitoring stopped.")
    except: pass

def on_close():
    global running, cap
    try:
        stop_audio_monitor()

        running = False
        if cap is not None and hasattr(cap, "isOpened") and cap.isOpened():
            cap.release()
        if status_list and status_list[-1] == 1 and times:
            times[-1] = (times[-1][0], datetime.now())
        save_log()
    except Exception:
        pass
    root.destroy()

def refresh_timeline():
    try:
        df = load_events()
        fig = timeline_figure(df, color_by_stage=False)
        fig_to_tk(fig, tl_img)
        status_bar.config(text="Timeline refreshed.")
    except Exception as e:
        messagebox.showerror("Timeline Error", str(e))

def refresh_stages():
    try:
        df = load_events()
        df = add_gaps_and_stage(df)
        fig = timeline_figure(df, color_by_stage=True)
        fig_to_tk(fig, st_img)
        status_bar.config(text="Stages estimated and visualized.")
    except Exception as e:
        messagebox.showerror("Stages Error", str(e))

def run_alerts(inactivity_limit=5, distress_threshold=1):
    alerts_box.delete("1.0", "end")
    try:
        df = load_events()
        df = df.sort_values("Start").reset_index(drop=True)
        for i in range(len(df)-1):
            gap = (df["Start"].iloc[i+1] - df["End"].iloc[i]).total_seconds()/60.0
            if gap > inactivity_limit:
                alerts_box.insert("end", f"⚠ Inactivity Alert: {gap:.2f} min idle after {df['End'].iloc[i]}\n")
            elif gap < distress_threshold:
                alerts_box.insert("end", f"‼ Distress Alert: frequent movement between {df['End'].iloc[i]} and {df['Start'].iloc[i+1]}\n")
            else:
                alerts_box.insert("end", f"✓ Normal Interval: {gap:.2f} min\n")
        alerts_box.insert("end", "\n--- Alerts complete ---\n")
        status_bar.config(text="Alerts generated.")
    except Exception as e:
        messagebox.showerror("Alerts Error", str(e))

def generate_daily_reports():
    try:
        df = load_events()
        groups = day_groups(df)
        if not groups:
            messagebox.showinfo("Reports", "No data available for report.")
            return

        for date_key, g in groups:
            staged = add_gaps_and_stage(g)
            summ = sleep_summary(staged)

            fig = timeline_figure(staged, color_by_stage=True)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)

            pdf_file = f"sleep_report_{date_key}.pdf"
            c = canvas.Canvas(pdf_file, pagesize=letter)
            W, H = letter

            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, H-50, "DAILY INFANT SLEEP REPORT")
            c.setFont("Helvetica", 12)
            c.drawString(50, H-90,  f"Date: {date_key}")
            c.drawString(50, H-115, f"Total Motion Events: {summ['events']}")
            c.drawString(50, H-140, f"Total Motion Duration: {summ['motion_min']:.2f} min")
            c.drawString(50, H-165, f"Estimated Sleep (quiet gaps ≥1m): {summ['sleep_min']:.2f} min")
            c.drawString(50, H-190, f"Monitoring Window: {summ['window_min']:.2f} min")
            c.drawString(50, H-215, f"Sleep Efficiency: {summ['sleep_eff_pct']:.1f}%")

            img = ImageReader(buf)
            c.drawImage(img, 50, 250, width=500, height=200, preserveAspectRatio=True, mask='auto')

            c.setFont("Helvetica-Oblique", 10)
            c.drawString(50, 60, "Generated by Smart Cradle System")
            c.showPage()
            c.save()

        status_bar.config(text="Daily reports generated.")
        messagebox.showinfo("Reports", "✅ Daily PDF report(s) generated in the app folder.")
    except Exception as e:
        messagebox.showerror("Report Error", str(e))

def view_db_records():
    try:
        conn = sqlite3.connect("smart_cradle.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, Start, End FROM motion_log ORDER BY id DESC LIMIT 100")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            messagebox.showinfo("Database Records", "No records found.")
            return

        popup = tk.Toplevel(root)
        popup.title("📊 Motion Log Records")
        popup.configure(bg=BG_COLOR)

        header = ["ID", "Start Time", "End Time"]
        for col, text in enumerate(header):
            lbl = tk.Label(
                popup, text=text,
                font=("Segoe UI", 12, "bold"),
                bg=PANEL_COLOR, fg=TITLE_COLOR,
                borderwidth=1, relief="solid", width=25
            )
            lbl.grid(row=0, column=col, sticky="nsew")

        for r, row in enumerate(rows, start=1):
            for c, val in enumerate(row):
                lbl = tk.Label(
                    popup, text=val,
                    font=("Segoe UI", 10),
                    bg="white", fg="black",
                    borderwidth=1, relief="solid", width=25
                )
                lbl.grid(row=r, column=c, sticky="nsew")

        popup.geometry("800x400")

    except Exception as e:
        messagebox.showerror("Database Error", str(e))

root = tk.Tk()
root.title("Smart Cradle — Monitor & Analytics")
root.configure(bg=BG_COLOR)
root.protocol("WM_DELETE_WINDOW", on_close)

title = tk.Label(root, text="Smart Cradle Monitoring", bg=BG_COLOR, fg=TITLE_COLOR, font=FONT_TITLE)
title.pack(pady=(12, 6))

controls = tk.Frame(root, bg=BG_COLOR)
controls.pack(fill="x", padx=12, pady=6)

def _start_webcam():
    start_monitoring("webcam")

def _start_file():
    start_monitoring("file")

btn_start_webcam = make_glass_button(controls, "Start Webcam", _start_webcam)
btn_start_webcam.pack(side="left", padx=6)

btn_start_file = make_glass_button(controls, "Start Video File", _start_file)
btn_start_file.pack(side="left", padx=6)

btn_stop = make_glass_button(controls, "Stop", stop_monitoring)
btn_stop.config(state="disabled")
btn_stop.pack(side="left", padx=6)

content = tk.Frame(root, bg=BG_COLOR)
content.pack(fill="both", expand=True, padx=12, pady=6)

video_panel = tk.Frame(content, bg=PANEL_COLOR, bd=0, highlightthickness=0)
video_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))

video_label = tk.Label(video_panel, bg=PANEL_COLOR)
video_label.pack(padx=10, pady=10, fill="both", expand=True)

analytics_frame = tk.Frame(content, bg=BG_COLOR)
analytics_frame.pack(side="right", fill="y")

notebook = ttk.Notebook(analytics_frame)
notebook.pack(fill="both", expand=True)

tab_timeline = tk.Frame(notebook, bg=PANEL_COLOR)
tab_stages   = tk.Frame(notebook, bg=PANEL_COLOR)
tab_reports  = tk.Frame(notebook, bg=PANEL_COLOR)
tab_alerts   = tk.Frame(notebook, bg=PANEL_COLOR)
tab_audio    = tk.Frame(notebook, bg=PANEL_COLOR) 

notebook.add(tab_timeline, text="Timeline")
notebook.add(tab_stages,   text="Stages")
notebook.add(tab_reports,  text="Reports")
notebook.add(tab_alerts,   text="Alerts")
notebook.add(tab_audio,    text="Audio")  

tl_img = tk.Label(tab_timeline, bg=PANEL_COLOR)
tl_img.pack(padx=10, pady=10)
tl_refresh_btn = make_glass_button(tab_timeline, "Refresh Timeline", refresh_timeline)
tl_refresh_btn.pack(pady=8)

st_img = tk.Label(tab_stages, bg=PANEL_COLOR)
st_img.pack(padx=10, pady=10)
st_refresh_btn = make_glass_button(tab_stages, "Estimate & Show Stages", refresh_stages)
st_refresh_btn.pack(pady=8)

rep_btn = make_glass_button(tab_reports, "Generate Daily PDF Reports", generate_daily_reports)
rep_btn.pack(pady=12, padx=10)

alerts_box = tk.Text(tab_alerts, height=16, bg="#0d1b2a", fg=TXT_COLOR, relief="flat", wrap="word")
alerts_box.pack(fill="both", expand=True, padx=10, pady=10)
alerts_btn = make_glass_button(tab_alerts, "Generate Alerts", run_alerts)
alerts_btn.pack(pady=8)

db_btn = make_glass_button(tab_reports, "View Database Records", view_db_records)
db_btn.pack(pady=12, padx=10)

audio_status = tk.Label(tab_audio, text="Cry: (stopped)", bg=PANEL_COLOR, fg="white",
                        font=("Segoe UI", 14, "bold"))
audio_status.pack(pady=(16, 8))

conf_label = tk.Label(tab_audio, text="Confidence: —", bg=PANEL_COLOR, fg=TXT_COLOR, font=("Segoe UI", 10))
conf_label.pack(pady=(0, 8))

def poll_cry_indicator():
    if audio_running:
        conf = float(np.mean(cry_conf_hist[-4:])) if cry_conf_hist else 0.0
        conf_label.config(text=f"Confidence: {conf:.2f}")
        if conf >= 0.6:
            audio_status.config(text="Cry: YES", fg="#ff8fa3")
        else:
            audio_status.config(text="Cry: NO", fg="white")
    else:
        audio_status.config(text="Cry: (stopped)", fg="white")
        conf_label.config(text="Confidence: —")
    audio_status.after(300, poll_cry_indicator)

poll_cry_indicator()

audio_start_btn = make_glass_button(tab_audio, "Start Cry Detection", start_audio_monitor)
audio_start_btn.pack(pady=6)

audio_stop_btn = make_glass_button(tab_audio, "Stop Cry Detection", stop_audio_monitor)
audio_stop_btn.pack(pady=6)

auto_soothe_var = tk.BooleanVar(value=True)
auto_chk = tk.Checkbutton(tab_audio, text="Auto-Soothe on Cry (launch Rocker)", variable=auto_soothe_var,
                          bg=PANEL_COLOR, fg=TXT_COLOR, selectcolor=PANEL_COLOR, activebackground=PANEL_COLOR,
                          activeforeground=TXT_COLOR, highlightthickness=0)
auto_chk.pack(pady=(10, 6))

def manual_soothe():
    launch_soothe()

soothe_btn = make_glass_button(tab_audio, "Soothe Now (Lullaby)", manual_soothe)
soothe_btn.pack(pady=6)

status_bar = tk.Label(root, text="Ready.", bg=BG_COLOR, fg=TXT_COLOR, anchor="w", font=("Segoe UI", 10))
status_bar.pack(fill="x", padx=12, pady=(0, 10))

root.bind("<q>", lambda e: on_close())

root.mainloop()