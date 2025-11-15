import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
df = pd.read_csv("sleep_log.csv")
df["Start"] = pd.to_datetime(df["Start"])
df["End"] = pd.to_datetime(df["End"])
# Create a plot
plt.figure(figsize=(12, 4))
for i, row in df.iterrows():
    plt.plot([row["Start"], row["End"]], [1, 1], linewidth=8)
# Formatting the plot
plt.title(" Infant Sleep Pattern - Motion Timeline")
plt.xlabel("Time")
plt.yticks([])
plt.grid(True, axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
# Save it as image and show plot
plt.savefig("sleep_pattern_graph.png")
plt.show()