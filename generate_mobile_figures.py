import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def setup_publication_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

def main():
    input_csv = 'Data Table - Sheet1.csv'
    output_dir = 'graphs_mobile'
    
    if not os.path.exists(input_csv):
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    if 'Input Tier' in df.columns:
        df['Input Tier'] = df['Input Tier'].ffill()
        df['Input Tier'] = df['Input Tier'].astype(str).str.strip()
    
    categorical_cols = ['Device', 'Input Tier', 'Model', 'Precision', 'Runtime']
    numeric_cols = [col for col in df.columns if col not in categorical_cols and not df[col].isnull().all()]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    setup_publication_style()

    tier_order = ["Short", "Medium", "Long"] 
    if 'Input Tier' in df.columns:
        actual_tiers = [t for t in tier_order if t in df['Input Tier'].unique()]
        if not actual_tiers:
            actual_tiers = sorted(df['Input Tier'].unique())
    else:
        actual_tiers = None

    device_colors = ["#2C3E50", "#8E44AD", "#27AE60", "#E67E22", "#C0392B"]
    precision_colors = ["#4F6D7A", "#A7C7E7"]

    for i, y_col in enumerate(numeric_cols, start=1):
        plt.figure(figsize=(7, 4.5))
        
        if 'Latency' in y_col and 'Input Tier' in df.columns and 'Device' in df.columns:
            ax = sns.barplot(data=df, x='Input Tier', y=y_col, hue='Device', palette=device_colors, order=actual_tiers, errorbar=None)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False, title='Android Device')
            plt.xlabel('Input Workload (Tier)')
        elif 'Model' in df.columns and 'Device' in df.columns:
            ax = sns.barplot(data=df, x='Device', y=y_col, hue='Model', palette=precision_colors, errorbar=None)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False, title='Model Precision')
            plt.xlabel('Physical Hardware')
        elif 'Device' in df.columns:
            ax = sns.barplot(data=df, x='Device', y=y_col, hue='Device', palette=device_colors, errorbar=None)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False, title='Device')
            plt.xlabel('Physical Hardware')
        else:
            plt.close()
            continue

        ax.set_ylim(bottom=0)
        plt.title("") 
        plt.ylabel(y_col)
        
        safe_name = y_col.split('(')[0].strip().replace(' ', '_').lower()
        plt.savefig(os.path.join(output_dir, f'mobile_figure_{i}_{safe_name}.png'), dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()