import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load all benchmark CSV files
files = glob.glob("benchmark_*.csv")
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

print("Loaded rows:", len(df))

# --------------------------------------------------
# 1. Overall summary (9 rows)
# --------------------------------------------------

overall = df.groupby(['model','runtime']).agg({
    'total_latency':'mean',
    'tokens_per_second':'mean',
    'memory_mb':'mean',
    'cpu_percent':'mean'
}).reset_index()

overall.to_csv("final_comparison_table.csv", index=False)
print("Saved final_comparison_table.csv")

# --------------------------------------------------
# 2. Detailed summary (27 rows)
# --------------------------------------------------

detailed = df.groupby(['model','runtime','category']).agg({
    'total_latency':'mean',
    'tokens_per_second':'mean',
    'memory_mb':'mean',
    'cpu_percent':'mean'
}).reset_index()

detailed.to_csv("detailed_results_by_length.csv", index=False)
print("Saved detailed_results_by_length.csv")

# --------------------------------------------------
# 3. Graphs
# --------------------------------------------------

os.makedirs("graphs", exist_ok=True)
sns.set_theme(style="whitegrid")

# Throughput
plt.figure(figsize=(10,6))
sns.barplot(data=overall, x="model", y="tokens_per_second", hue="runtime")
plt.title("Throughput Comparison")
plt.ylabel("Tokens per Second")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("graphs/throughput.png", dpi=300)
plt.close()

# Memory
plt.figure(figsize=(10,6))
sns.barplot(data=overall, x="model", y="memory_mb", hue="runtime")
plt.title("Memory Usage Comparison")
plt.ylabel("Memory (MB)")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("graphs/memory.png", dpi=300)
plt.close()

# Latency scaling (input length)
plt.figure(figsize=(10,6))
sns.barplot(data=detailed, x="category", y="total_latency", hue="runtime")
plt.title("Latency Scaling by Input Length")
plt.ylabel("Latency (seconds)")
plt.xlabel("Input Category")
plt.tight_layout()
plt.savefig("graphs/latency_scaling.png", dpi=300)
plt.close()

print("Graphs saved in /graphs folder")