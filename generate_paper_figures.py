import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os

COLORS = {
    "PyTorch (FP32)": "#2C3E50",
    "ONNX (FP32)": "#4F6D7A",
    "ONNX (INT8)": "#A7C7E7"
}

# EXACTLY 3 models expected.
EXPECTED_MODELS = ["t5-small", "distilbart-6-6", "bart-base"] 

def setup_publication_style():
    sns.set_style("white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

def generate_scientific_bar_chart(df, y_col, y_label, output_filename):
    plt.figure(figsize=(6, 4))
    
    clean_runtimes = {
        "pytorch_fp32": "PyTorch (FP32)",
        "onnx_fp32": "ONNX (FP32)",
        "onnx_int8": "ONNX (INT8)"
    }
    df["clean_runtime"] = df["runtime"].map(clean_runtimes)
    
    clean_models = {
        "t5-small": "T5-Small",
        "distilbart-6-6": "DistilBART (6-6)",
        "bart-base": "BART-Base"
    }
    df["clean_model"] = df["model"].map(clean_models)
    
    ordered_clean_models = [clean_models[m] for m in EXPECTED_MODELS]
    ordered_clean_runtimes = ["PyTorch (FP32)", "ONNX (FP32)", "ONNX (INT8)"]

    ax = sns.barplot(
        data=df,
        x="clean_model",     
        y=y_col,
        hue="clean_runtime", 
        palette=COLORS,
        order=ordered_clean_models, 
        hue_order=ordered_clean_runtimes, 
        errorbar=None          
    )
    
    ax.set_ylim(bottom=0)
    plt.xlabel("Model Architecture")
    plt.ylabel(y_label)
    plt.title("")
    
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="Inference Engine")
    
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Artifact successfully generated: {output_filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="final_comparison_table.csv")
    parser.add_argument("--output-dir", type=str, default="graphs")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Fatal Error: Benchmark file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    
    if "Model" in df.columns:
        column_mapping = {
            "Model": "model",
            "Runtime": "runtime",
            "Avg Latency (s)": "Avg_Latency_sec",
            "Tokens/sec": "Tokens_Per_Sec",
            "Avg Memory (MB)": "Avg_Memory_MB"
        }
        df.rename(columns=column_mapping, inplace=True)
    
    # This line forcefully deletes anything that isn't our 3 core models
    df = df[df["model"].isin(EXPECTED_MODELS)].copy()

    setup_publication_style()
    
    generate_scientific_bar_chart(df, "Avg_Latency_sec", "Latency (Seconds)", os.path.join(args.output_dir, "figure_1_latency.png"))
    generate_scientific_bar_chart(df, "Tokens_Per_Sec", "Tokens per Second", os.path.join(args.output_dir, "figure_2_throughput.png"))
    generate_scientific_bar_chart(df, "Avg_Memory_MB", "Memory Usage (MB)", os.path.join(args.output_dir, "figure_3_memory.png"))

if __name__ == "__main__":
    main()