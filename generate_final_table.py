import pandas as pd
import glob

csv_files = glob.glob("benchmark_*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

summary = combined_df.groupby(["model", "runtime"]).agg(
    Avg_Latency_sec=("total_latency", "mean"),
    Tokens_Per_Sec=("tokens_per_second", "mean"),
    Avg_Memory_MB=("memory_mb", "mean"),
    Avg_CPU_Percent=("cpu_percent", "mean")
).reset_index()

summary["Avg_Latency_sec"] = summary["Avg_Latency_sec"].round(2)
summary["Tokens_Per_Sec"] = summary["Tokens_Per_Sec"].round(2)
summary["Avg_Memory_MB"] = summary["Avg_Memory_MB"].round(2)
summary["Avg_CPU_Percent"] = summary["Avg_CPU_Percent"].round(2)

print("--- Final Results ---")
print(summary.to_string(index=False))

summary.to_csv("final_comparison_table.csv", index=False)
print("\nSaved to final_comparison_table.csv")