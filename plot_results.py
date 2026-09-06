"""
Generate poster-quality comparison plots from evaluation CSVs.

Reads `policy_evaluation_summary.csv` and `policy_timeline_data.csv` and
produces styled, high-resolution (300 DPI) bar and line charts comparing
LRU/LFU/FIFO/RANDOM/RL eviction policies.

SCHEMA (kept in sync with evaluate_policies.py): this script expects
`Workload` (timeline) and `Workload_Set`, `Policy`, `Total_Runtime(s)`,
`P99_Latency(ms)`, `P50_Latency(ms)`, `Final_Cache_Hit_Ratio(%)`
(summary). `evaluate_policies.py` runs three workload scenarios
(Trained_Workload, Moving_Workload, Unseen_Workload) across five
policies and emits exactly these columns — if you change either
script's column names, update the other to match.

Inputs: policy_evaluation_summary.csv, policy_timeline_data.csv
    (both read from the current working directory).
Outputs: evaluation_plots_poster/*.png (timeline plots per workload,
    latency bar charts, generalization-improvement chart, final hit
    ratio chart).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
RESULTS_DIR = "evaluation_plots_poster"
SUMMARY_FILE = "policy_evaluation_summary.csv"
TIMELINE_FILE = "policy_timeline_data.csv"
POLICIES = ["LRU", "LFU", "FIFO", "RANDOM", "RL"]

# --- Styling Constants for Poster ---
FONT_TITLE = 36
FONT_LABEL = 30
FONT_TICK = 26
FONT_LEGEND = 24

# Setup Output Directory
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Set Seaborn theme
sns.set_theme(style="whitegrid")
# Ensure high resolution for print
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# Bold text for better readability from a distance
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


def format_workload_name(name: str) -> str:
    """Strip underscores and the literal "Workload" suffix for display (e.g. "Moving_Workload" -> "Moving")."""
    if isinstance(name, str):
        return name.replace("_", " ").replace("Workload", "").strip()
    return name


def save_plot(filename: str) -> None:
    """Save the current matplotlib figure into RESULTS_DIR and close it.

    Args:
        filename: Output filename (joined with RESULTS_DIR).
    """
    path = os.path.join(RESULTS_DIR, filename)
    # bbox_inches='tight' trims all extra whitespace around the chart
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved {filename}")


# --- Plotting Functions ---

def plot_hit_ratio_timeline_separated(df_timeline: pd.DataFrame) -> None:
    """
    Render one hit-ratio-over-time line chart per distinct workload in
    `df_timeline['Workload']`, saved as timeline_<workload>.png.

    Special-cases a workload literally named "Moving_Workload" by
    drawing a vertical "Phase Shift" marker at a hardcoded query index
    (10.5) — this assumes that specific workload always shifts phase at
    that fixed point; update the constant if the workload generator changes.

    Args:
        df_timeline: Must contain columns Workload, Query_Index,
            Cache_Hit_Ratio, Policy (see module docstring for the
            current schema mismatch with evaluate_policies.py's output).
    """
    print("Generating Hit Ratio Timeline Plots...")

    palette = sns.color_palette("viridis", n_colors=len(POLICIES))
    policy_color_map = dict(zip(POLICIES, palette))
    unique_workloads = df_timeline['Workload'].unique()

    for workload in unique_workloads:
        plt.figure(figsize=(16, 8))

        subset = df_timeline[df_timeline['Workload'] == workload]

        sns.lineplot(
            data=subset,
            x='Query_Index',
            y='Cache_Hit_Ratio',
            hue='Policy',
            palette=policy_color_map,
            marker='o',
            markersize=9,
            linewidth=4,
            hue_order=POLICIES
        )

        y_min = max(subset['Cache_Hit_Ratio'].min() - 5, 0)
        y_max = min(subset['Cache_Hit_Ratio'].max() + 5, 100)
        plt.ylim(y_min, y_max)

        if workload == "Moving_Workload":
            plt.axvline(x=10.5, color='red', linestyle='--', linewidth=3, alpha=0.8)
            plt.text(10.6, y_min + 2, "Phase Shift", color='red', fontsize=22, fontweight='bold')

        clean_name = format_workload_name(workload)
        plt.title(f"Adaptability: {clean_name}", fontsize=FONT_TITLE, pad=15)
        plt.xlabel("Query Sequence", fontsize=FONT_LABEL)
        plt.ylabel("Hit Ratio (%)", fontsize=FONT_LABEL)

        plt.xticks(fontsize=FONT_TICK)
        plt.yticks(fontsize=FONT_TICK)

        plt.legend(
            title='Policy',
            title_fontsize=FONT_LEGEND,
            fontsize=FONT_LEGEND,
            bbox_to_anchor=(1.01, 1),
            loc='upper left',
            borderaxespad=0,
            frameon=True
        )

        save_plot(f"timeline_{workload.lower()}.png")


def plot_latency_comparison_combined(df_summary: pd.DataFrame) -> None:
    """
    Render grouped bar charts of P99 and P50 latency (ms) by workload
    scenario and policy, saved as bar_combined_p99.png / bar_combined_p50.png.

    Args:
        df_summary: Must contain columns Workload_Set, Policy,
            'P99_Latency(ms)', 'P50_Latency(ms)'.
    """
    print("Generating Latency Bar Charts...")

    metrics = {
        'P99_Latency(ms)': 'Tail Latency (P99): Worst-Case Stability',
        'P50_Latency(ms)': 'Median Latency (P50): Typical Performance'
    }

    df_plot = df_summary.copy()
    df_plot['Workload_Set'] = df_plot['Workload_Set'].apply(format_workload_name)

    for metric_col, metric_title in metrics.items():
        plt.figure(figsize=(15, 8))

        # Removed edgecolor and linewidth for a cleaner, flatter look
        sns.barplot(
            data=df_plot,
            x='Workload_Set',
            y=metric_col,
            hue='Policy',
            hue_order=POLICIES,
            palette='viridis'
        )

        plt.title(metric_title, fontsize=FONT_TITLE, pad=15)
        plt.xlabel("Workload Scenario", fontsize=FONT_LABEL)
        plt.ylabel("Latency (ms)", fontsize=FONT_LABEL)

        plt.xticks(fontsize=FONT_TICK)
        plt.yticks(fontsize=FONT_TICK)

        plt.legend(
            title='Policy',
            title_fontsize=FONT_LEGEND,
            fontsize=FONT_LEGEND,
            bbox_to_anchor=(1.01, 1),
            loc='upper left'
        )

        # Removed bar_label calls to reduce clutter
        save_plot(f"bar_combined_{metric_col.split('_')[0].lower()}.png")


def plot_generalization_summary(df_summary: pd.DataFrame) -> None:
    """
    Compute each policy's total-runtime improvement over LRU (%) per
    workload scenario, and render as a grouped bar chart
    (bar_generalization_improvement.png). No-ops (returns without
    plotting) if the 'LRU' column is missing after pivoting, or if the
    resulting improvement table is empty.

    Args:
        df_summary: Must contain columns Workload_Set, Policy,
            'Total_Runtime(s)'.
    """
    print("Generating Generalization Summary...")

    df_calc = df_summary.copy()
    df_calc['Total_Runtime(s)'] = pd.to_numeric(df_calc['Total_Runtime(s)'], errors='coerce')
    df_calc['Workload_Set'] = df_calc['Workload_Set'].astype(str).str.strip()

    pivot_df = df_calc.pivot(index='Workload_Set', columns='Policy', values='Total_Runtime(s)')

    if 'LRU' not in pivot_df.columns: return

    improvement_df = pd.DataFrame()
    for policy in pivot_df.columns:
        if policy == 'LRU': continue
        vals = ((pivot_df['LRU'] - pivot_df[policy]) / pivot_df['LRU']) * 100
        temp = pd.DataFrame({'Workload_Set': pivot_df.index, 'Policy': policy, 'Improvement': vals.values})
        improvement_df = pd.concat([improvement_df, temp])

    if improvement_df.empty: return

    improvement_df['Workload_Set'] = improvement_df['Workload_Set'].apply(format_workload_name)

    plt.figure(figsize=(15, 8))

    # Removed edgecolor and linewidth
    sns.barplot(
        data=improvement_df,
        x='Workload_Set',
        y='Improvement',
        hue='Policy',
        palette='viridis'
    )

    plt.axhline(0, color='black', linewidth=2)
    plt.title("Runtime Improvement vs. LRU (%)", fontsize=FONT_TITLE, pad=15)
    plt.xlabel("Workload Scenario", fontsize=FONT_LABEL)
    plt.ylabel("% Faster than LRU", fontsize=FONT_LABEL)

    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)

    plt.legend(
        title='Policy',
        title_fontsize=FONT_LEGEND,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(1.01, 1),
        loc='upper left'
    )

    # Removed bar_label calls
    save_plot("bar_generalization_improvement.png")


def plot_final_hit_ratio_combined(df_summary: pd.DataFrame) -> None:
    """
    Render a grouped bar chart of final cumulative hit ratio (%) by
    workload scenario and policy, saved as bar_final_hit_ratio_combined.png.

    Args:
        df_summary: Must contain columns Workload_Set, Policy,
            'Final_Cache_Hit_Ratio(%)'.
    """
    print("Generating Final Hit Ratio Bar Chart...")

    df_plot = df_summary.copy()
    df_plot['Workload_Set'] = df_plot['Workload_Set'].apply(format_workload_name)

    plt.figure(figsize=(15, 8))

    # Removed edgecolor and linewidth
    sns.barplot(
        data=df_plot,
        x='Workload_Set',
        y='Final_Cache_Hit_Ratio(%)',
        hue='Policy',
        hue_order=POLICIES,
        palette='viridis'
    )

    plt.title("Cumulative Hit Ratio Efficiency", fontsize=FONT_TITLE, pad=15)
    plt.xlabel("Workload Scenario", fontsize=FONT_LABEL)
    plt.ylabel("Hit Ratio (%)", fontsize=FONT_LABEL)

    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.ylim(0, 105)

    plt.legend(
        title='Policy',
        title_fontsize=FONT_LEGEND,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(1.01, 1),
        loc='upper left'
    )

    # Removed bar_label calls
    save_plot("bar_final_hit_ratio_combined.png")


# --- Main Execution ---

def main() -> None:
    """Load both evaluation CSVs and generate all poster plots, or print a friendly error if missing."""
    try:
        df_summary = pd.read_csv(SUMMARY_FILE)
        df_timeline = pd.read_csv(TIMELINE_FILE)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Required file not found: {e.filename}")
        print("Please run evaluate_policies.py first.")
        return

    # Generate all plots
    plot_hit_ratio_timeline_separated(df_timeline)
    plot_latency_comparison_combined(df_summary)
    plot_generalization_summary(df_summary)
    plot_final_hit_ratio_combined(df_summary)

    print(f"\nAll poster plots generated in '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()