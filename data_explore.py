import matplotlib.pyplot as plt
import pandas as pd

# Load timeline data
df = pd.read_csv("policy_timeline_data.csv")

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

# ---- Cache Hit Ratio ----
for policy in df["Policy"].unique():
    sub = df[df["Policy"] == policy]
    axes[0].plot(sub["Query_Index"], sub["Cache_Hit_Ratio"], label=policy)

axes[0].set_ylabel("Cache Hit Ratio (%)")
axes[0].set_title("Cache Hit Ratio Over Time")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ---- Query Latency ----
for policy in df["Policy"].unique():
    sub = df[df["Policy"] == policy]
    axes[1].plot(sub["Query_Index"], sub["Query_Latency_ms"], label=policy)

axes[1].set_xlabel("Query Index")
axes[1].set_ylabel("Query Latency (ms)")
axes[1].set_title("Query Latency Over Time")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("my_plot_1.png")
plt.show()
