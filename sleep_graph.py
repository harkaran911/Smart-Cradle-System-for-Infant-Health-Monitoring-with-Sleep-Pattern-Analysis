import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
df = pd.read_csv("sleep_log.csv")
df["Start"] = pd.to_datetime(df["Start"])
df["End"] = pd.to_datetime(df["End"])

# Plot the sleep/wake durations
plt.figure(figsize=(10, 4))
for i in df.index:
    plt.plot([df["Start"][i], df["End"][i]], [i, i], color="blue", linewidth=6)

plt.title("Infant Motion Activity Timeline")
plt.xlabel("Time")
plt.ylabel("Motion Events")
plt.tight_layout()
plt.grid(True)
plt.show()
