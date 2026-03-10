import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set academic plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Load the data
df = pd.read_csv("final_comparison_table.csv")

# Ensure runtimes are ordered logically for the charts
runtime_order = ["pytorch_fp32", "onnx_fp32", "onnx_int8"]
df['runtime'] = pd.Categorical(df['runtime'], categories=runtime_order, ordered=True)

# 1. Latency Figure
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="model", y="Avg_Latency_sec", hue="runtime", palette="Blues_d")
plt.title("Average Inference Latency per Article")
plt.ylabel("Latency (Seconds)")
plt.xlabel("Model Architecture")
plt.legend(title="Runtime Configuration")
plt.tight_layout()
plt.savefig("figure_1_latency.png", dpi=300)
plt.close()

# 2. Throughput Figure
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="model", y="Tokens_Per_Sec", hue="runtime", palette="Greens_d")
plt.title("Generation Throughput")
plt.ylabel("Tokens per Second")
plt.xlabel("Model Architecture")
plt.legend(title="Runtime Configuration")
plt.tight_layout()
plt.savefig("figure_2_throughput.png", dpi=300)
plt.close()

# 3. Memory Consumption Figure
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="model", y="Avg_Memory_MB", hue="runtime", palette="Reds_d")
plt.title("Average Memory Footprint During Inference")
plt.ylabel("Memory Usage (MB)")
plt.xlabel("Model Architecture")
plt.axhline(y=4096, color='r', linestyle='--', label='4GB RAM Limit') # Visual constraint line
plt.legend(title="Runtime Configuration")
plt.tight_layout()
plt.savefig("figure_3_memory.png", dpi=300)
plt.close()

print("Publication figures successfully generated and saved as PNGs.")