import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from cmcrameri import cm
from collections import defaultdict
import glob
import os 
import numpy as np
import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Optional

from knowledge_drift_plot import get_equally_spaced_colors

def plot_comparison_charts():
    # Apply plotting parameters
    seaborn_params = {
        "style": "white",
        "palette": get_equally_spaced_colors(cm.roma, 6, start=0.1, end=0.9),
        "context": "talk",
        "color_codes": True,
    }

    matplotlib_params = {
        "figure.figsize": (12, 4),
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
    }
    
    plt.rcParams.update(matplotlib_params)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)  # sharey=True gives single y-axis
    
    # ===== ASSIGN YOUR DATA HERE =====
    datasets = ["CNN/DailyMail", "Multi-News", "GovReport"]
    
    # Data for each dataset: [model1, model2, model3]
    # Each model has [ROUGE-L, BLEU, BERTScore, Embed Sim, Coverage]
    
    # CNN/DailyMail data
    cnn_data = {
        "0.0prune_kdtrue": [0.1397*5, 0.0176*10, 0.8390, 0.7191, 0.8025],  # Multiply ROUGE-L and BLEU by 10
        "50.0prune_kdfalse": [0.046*5, 0.0051*10, 0.7755, 0.6230, 0.6583],
        "0.50% Sparsity Drift": [0.122*5, 0.0144*10, 0.8322, 0.7029, 0.7692],
    }
    cnn_triggers = 99
    cnn_total_tokens = 9174 * 5
    
    # Multi-News data
    multinews_data = {
        "Dense": [0.118*5, 0.0231*10, 0.7859, 0.6552, 0.5033],
        "50% Sparsity": [0.063*5, 0.0060*10, 0.7406, 0.4640, 0.1905],
        "0.50% Sparsity Drift": [0.099*5, 0.0168*10, 0.7690, 0.5747, 0.3864],
    }
    multi_news_triggers = 98
    multi_news_tokens = 18970 * 5

    
    # GovReport data
    govreport_data = {
        "Dense": [0.172*5, 0.0409*10, 0.7954, 0.7128, 0.5335],
        "50% Sparsity": [0.123*5, 0.0163*10, 0.7637, 0.5894, 0.2967],
        "50% Sparsity w/ Drift enabled": [0.155*5, 0.0261*10, 0.7788, 0.6367, 0.3949],
    }
    govreport_triggers = 97
    govreport_tokens = 19549 * 5
    
    all_data = [cnn_data, multinews_data, govreport_data]
    # ===== END DATA ASSIGNMENT =====
    
    metrics = ["ROUGE-L\n(×5)", "BLEU\n(×10)", "BERTScore\n(F1)", "Embed\nSim", "Coverage"]
    spaced_colors = get_equally_spaced_colors(cm.lipari, 3, start=0.1, end=0.8)
    
    bar_width = 0.25
    x = np.arange(len(metrics))
    
    for idx, (ax, data, dataset_name) in enumerate(zip(axes, all_data, datasets)):
        model_names = list(data.keys())
        
        for i, model_name in enumerate(model_names):
            offset = (i - 1) * bar_width
            ax.bar(
                x + offset,
                data[model_name],
                bar_width,
                label=model_name,
                color=spaced_colors[i],
                alpha=0.8
            )
        
        subtitle = ""
        if "drift" in str(data.keys()).lower():
            if idx == 0:  # CNN
                subtitle = f"\n({cnn_triggers} retriggers / {cnn_total_tokens:,} tokens)"
            elif idx == 1:  # Multi-News
                subtitle = f"\n({multi_news_triggers} retriggers / {multi_news_tokens:,} tokens)"
            elif idx == 2:  # GovReport
                subtitle = f"\n({govreport_triggers} retriggers / {govreport_tokens:,} tokens)"

        ax.set_title(dataset_name, fontsize=12, fontweight='bold')
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
                ha='center', fontsize=8, fontweight='normal', style='italic')
        # ax.set_title(dataset_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.grid(axis='y', alpha=0.6, linestyle='--')
        
        # Only show y-label on leftmost plot
        if idx == 0:
            ax.set_ylabel('Score', fontsize=11)
        
        # Only show legend on rightmost plot - positioned in top-right corner
        # if idx == 2:
        #     ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Add legend at the bottom center of the whole figure
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=len(labels), fontsize=10, frameon=False)
    
    plt.tight_layout()
    dump_dir = "/users/grad/abhishektyagi/wanda/wanda/results/plots/knowledge_drift_standard_datasets"
    os.makedirs(dump_dir, exist_ok=True)
    out_path = os.path.join(dump_dir, "comparison_charts.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison charts: {out_path}")

if __name__ == "__main__":
    plot_comparison_charts()