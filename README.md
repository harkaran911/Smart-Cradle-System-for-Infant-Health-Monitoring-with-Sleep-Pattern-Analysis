🌙 Smart Cradle System for Infant Health Monitoring
with Sleep Pattern Analysis (Software-Based Computer Vision System)

A fully software-based intelligent infant monitoring solution that uses computer vision, data analytics, and a Tkinter GUI to monitor infant movement, analyze sleep quality, generate reports, visualize patterns, and simulate cradle automation — no hardware required.

Designed for parents, hospitals, neonatal units (NICU), and researchers seeking a low-cost, non-intrusive, and stress-free monitoring system.

🚀 Key Features
🔹 1. Real-Time Motion Detection (OpenCV)
• Detects infant movement using background subtraction.
• Highlights motion areas with bounding boxes.
• Logs every motion start and end time.

🔹 2. GUI Application (Tkinter)
• Beautiful, user-friendly interface allowing:
• Live monitoring
• Video file analysis
• One-click reports
• Alerts & rocking simulation
• Sleep pattern graphs
• SQLite database viewing

🔹 3. Sleep Pattern & Timeline Analysis
• Generates a timeline graph of motion episodes.
• Shows sleep disruption patterns.
• Helps identify deep sleep vs active sleep.

🔹 4. Automated Reports
• Daily sleep reports (total motion, average duration).
• Exportable summaries for doctors/parents.

🔹 5. Alert Simulation
• Distress alerts (frequent movement)
• Inactivity alerts (no movement for long period)

🔹 6. Cradle Rocking Simulation with Audio
• Plays gentle rocking sound for 5 seconds per event.
• Demonstrates automation behavior.

🔹 7. SQLite Data Logging
• Saves all motion timestamps to a local database.
• Allows retrieving historical data.

🧩 Tech Stack

| Component                | Technology     |
| ------------------------ | -------------- |
| Programming Language     | Python 3.x     |
| Video Processing         | OpenCV         |
| GUI Framework            | Tkinter        |
| Data Handling            | Pandas, SQLite |
| Plotting & Visualization | Matplotlib     |
| Audio Playback           | pygame         |
| Image Handling           | Pillow         |

🖥️ How to Run the Project

1. Clone the Repository
     git clone https://github.com/harkaran911/Smart-Cradle-System-for-Infant-Health-Monitoring-with-Sleep-Pattern-Analysis.git
     cd Smart-Cradle-System

2. Install Dependencies
     pip install -r requirements.txt

3. Run the GUI
     python smart_cradle_gui.py

🧪 How the System Works

Step 1 — Input Selection
User chooses Live Webcam Feed or Video File Input
The system begins real-time or offline analysis

Step 2 — Motion Detection
Converts frames to grayscale
Applies Gaussian blur
Detects difference between background and current frame
Identifies motion with contours
Logs timestamps

Step 3 — Data Storage
Every motion event is saved in:
sleep_log.csv
smart_cradle.db (SQLite table)

Step 4 — Analysis Tools
Graphs
Sleep pattern timeline
Stage estimation
Alerts
Reports

Step 5 — Rocking Simulation
Plays gentle rocking sound for 5 seconds.

🏥 Use Cases

🍼 Parents at Home
1. Low-cost infant sleep tracking
2. Lightweight laptop/mobile monitoring

🏥 Hospitals & NICUs
1. Non-contact monitoring reduces stress
2. Useful when sensors are not advisable
3. Software can integrate with CCTV feeds

🎓 Academic & Research
1. Perfect for infant behavior studies
2. Excellent for machine learning datasets

📜 License

MIT License © 2025 Smart Cradle Development Team
