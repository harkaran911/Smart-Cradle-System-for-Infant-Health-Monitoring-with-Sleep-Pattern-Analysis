import pandas as pd
import time
from datetime import datetime
import pygame
import tkinter as tk
from tkinter import messagebox

pygame.mixer.init()

sound_file = "gentle-rocking-lullaby.wav"

try:
    df = pd.read_csv("sleep_log.csv")
except FileNotFoundError:
    root = tk.Tk()
    root.withdraw()  # hide main window
    messagebox.showerror("Error", "Motion log not found. Please run monitoring first.")
    exit()

# Start popup
root = tk.Tk()
root.withdraw()  # hide main window
messagebox.showinfo("Cradle Rocker", "🎵 Cradle rocking started with lullaby!")

for index, row in df.iterrows():
    start_time = row['Start']
    end_time = row['End']

    # Motion duration in seconds
    motion_duration = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds()

    # Rocking time decision
    rocking_time = 5 if motion_duration < 10 else 10

    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play(-1)

    for i in range(rocking_time):
        time.sleep(1)

    pygame.mixer.music.stop()

# End popup
messagebox.showinfo("Cradle Rocker", "✅ Cradle rocking complete.")
