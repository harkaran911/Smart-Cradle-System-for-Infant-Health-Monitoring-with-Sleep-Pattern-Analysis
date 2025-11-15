import pandas as pd
from datetime import datetime
df = pd.read_csv("sleep_log.csv")
df["Start"] = pd.to_datetime(df["Start"])
df["End"] = pd.to_datetime(df["End"])
# Sort just in case
df.sort_values("Start", inplace=True)
# Thresholds
inactivity_limit = 5  # minutes
distress_threshold = 1  # minutes
# Simulate alert generation
print("\n ALERT SYSTEM ACTIVATED \n")

for i in range(len(df) - 1):
    gap = (df.iloc[i + 1]["Start"] - df.iloc[i]["End"]).total_seconds() / 60
    if gap > inactivity_limit:
        print(f"Inactivity Alert: No motion detected for {gap:.2f} minutes after {df.iloc[i]['End']}")
    elif gap < distress_threshold:
        print(f"Distress Alert: Frequent movement detected between {df.iloc[i]['End']} and {df.iloc[i+1]['Start']}")
    else:
        print(f"Normal Interval between {df.iloc[i]['End']} and {df.iloc[i+1]['Start']}")
print("\n--- All alerts simulated successfully ---")