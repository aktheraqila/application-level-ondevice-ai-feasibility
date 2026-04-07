import pandas as pd
import glob
import sys

def main():
    csv_files = glob.glob("benchmark_*.csv")
    if not csv_files:
        print("Error: No benchmark CSV files found.")
        sys.exit(1)

    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    # Clean whitespace and unify names
    combined_df["model"] = combined_df["model"].astype(str).str.strip()
    combined_df["model"] = combined_df["model"].replace({
        "facebook/bart-base": "bart-base",
        "bart-base ": "bart-base"
    })

    # Group and calculate
    summary = combined_df.groupby(["model", "runtime"]).agg(
        Avg_Latency_sec=("total_latency", "mean"),
        Tokens_Per_Sec=("tokens_per_second", "mean"),
        Avg_Memory_MB=("memory_mb", "mean"),
        Avg_CPU_Percent=("cpu_percent", "mean")
    ).reset_index()

    # Rounding
    for col in ["Avg_Latency_sec", "Tokens_Per_Sec", "Avg_Memory_MB", "Avg_CPU_Percent"]:
        summary[col] = summary[col].round(2)

    print("--- Final Results ---")
    print(summary.to_string(index=False))

    summary.to_csv("final_comparison_table.csv", index=False)
    print("\nSaved to final_comparison_table.csv")

if __name__ == "__main__":
    main()