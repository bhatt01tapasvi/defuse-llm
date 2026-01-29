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


def truncate_colormap(cmap, minval=0.1, maxval=0.9, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def is_dark_color(color):
    # Convert the color to RGB
    r, g, b = color[:3]
    # Calculate the luminance using a formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    # Return True if the color is dark, False otherwise
    return luminance < 0.5


def get_equally_spaced_colors(colormap, n, start=0.15, end=0.85):
    # Generating 'n' equally spaced values within the specified range
    values = np.linspace(start, end, n)

    # Retrieving colors from the colormap
    colors = [colormap(val) for val in values]

    return colors


def hex_to_rgba(hex_color):
    """Convert a hex color string to a normalized RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def display_color_swatches(input_colors, n=10, is_colormap=False):
    """
    Displays color swatches for 'n' equally spaced colors from the given colors.
    """
    if is_colormap:
        # Get 'n' equally spaced colors from the colormap
        colors = get_equally_spaced_colors(input_colors, n)
    else:
        colors = input_colors

    # Create a figure and a subplot
    fig, ax = plt.subplots(figsize=(n, 1))

    # Create a bar for each color
    for i, color in enumerate(colors):
        rect = patches.Rectangle(
            (i, 0), 1, 1, linewidth=1, edgecolor="none", facecolor=color
        )
        ax.add_patch(rect)

    # Set limits and turn off axes
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.show()


def plot_bar_color_test():
    fig, ax = plt.subplots()
    spaced_colors = get_equally_spaced_colors(cm.roma, 3, start=0.1, end=0.0)
    ax.bar(
        ["A", "B", "C"],
        [1, 2, 3],
        color=[spaced_colors[0], spaced_colors[1], spaced_colors[2]],
    )
    plt.tight_layout()
    dump_dir = os.path.join("results", "plots", "dump")
    os.makedirs(dump_dir, exist_ok=True)
    out_path = os.path.join(dump_dir, "bar_color_test.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved bar color test plot: {out_path}")


def plot_line_test():
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(np.random.rand(10), label=f"Line {i+1}")
    ax.set_title("Random Line Test", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Axis", fontsize=12)
    ax.set_ylabel("Y Axis", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, loc="best", fontsize=10)
    plt.tight_layout()
    dump_dir = os.path.join("results", "plots", "dump")
    os.makedirs(dump_dir, exist_ok=True)
    out_path = os.path.join(dump_dir, "line_test.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved line test plot: {out_path}")


##############################################################################
# SET UP NAMED COLORS
##############################################################################

basic_colors_dict = {
    "red": "#ff8a67",
    "blue": "#32bcdd",
    "yellow": "#ffc24c",
    "teal": "#80e2d8",
    "green": "#b7e5ca",
    "oat": "#fdfae9",
    "light_grey": "#f2f2f2",
    "mid_grey": "#dfdfdf",
    "darkgrey": "#453e3a",
}

roma_10 = {
    0: (0.633994, 0.38467, 0.110736, 1.0),
    1: (0.69512, 0.510297, 0.171371, 1.0),
    2: (0.761963, 0.652786, 0.276547, 1.0),
    3: (0.818865, 0.805562, 0.471253, 1.0),
    4: (0.801918, 0.905092, 0.688973, 1.0),
    5: (0.687217, 0.911341, 0.811442, 1.0),
    6: (0.49418, 0.835884, 0.843605, 1.0),
    7: (0.308653, 0.709514, 0.813746, 1.0),
    8: (0.199968, 0.578679, 0.763292, 1.0),
    9: (0.146549, 0.45526, 0.710977, 1.0),
}

# Make this mapping in case we want matched colors in different plots
demo2color = {
    "conversation_type": {
        "controversy": (0.005193, 0.098238, 0.349842),
        "values": (0.981354, 0.800406, 0.981267),
        "unguided": (0.511253, 0.510898, 0.193296),
    },
    "gender": {
        "Non-binary": (0.180627, 0.129916, 0.300244),
        "Female": (0.900472, 0.900123, 0.940051),
        "Male": (0.763197, 0.428302, 0.605491),
    },
    "age": {
        "65+": (0.999831, 0.999745, 0.799907),
        "55-64": (0.100227, 0.100908, 0.003791),
        "45-54": (0.872057, 0.460829, 0.311198),
        "35-44": (0.498095, 0.233152, 0.205757),
        "18-24": (0.948022, 0.763539, 0.374977),
        "25-34": (0.284955, 0.166067, 0.106101),
    },
    "ethnicity_simplified": {
        "Other": (0.01137, 0.07324, 0.148284),
        "Mixed": (0.992307, 0.959017, 0.856609),
        "Asian": (0.639835, 0.384073, 0.40417),
        "Hispanic": (0.912931, 0.60098, 0.450565),
        "Black": (0.320355, 0.358086, 0.480032),
        "White": (0.848263, 0.430214, 0.369804),
    },
    "religion_simplified": {
        "Other": (0.101441, 0.20011, 0.700194),
        "Muslim": (1.0, 0.999989, 0.400094),
        "Jewish": (0.325731, 0.523075, 0.498902),
        "Christian": (0.567196, 0.76625, 0.434306),
        "No Affiliation": (0.188513, 0.368188, 0.614982),
    },
    "location_special_region": {
        "Middle East": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        "N. America": (1.0, 0.4980392156862745, 0.054901960784313725),
        "Asia": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
        "Africa": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
        "Aus & NZ": (
            0.5803921568627451,
            0.403921568627451,
            0.7411764705882353,
        ),
        "Latam": (
            0.5490196078431373,
            0.33725490196078434,
            0.29411764705882354,
        ),
        "UK": (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
        "Europe": (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
        "US": (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    },
}

# Nice clean mapping for plot axis labels
demo2label = {
    "conversation_type": {
        "controversy guided": "Controversy",
        "values guided": "Values",
        "unguided": "Unguided",
    },
    "gender": {
        "Non-binary / third gender": "Non-binary",
        "Female": "Female",
        "Male": "Male",
    },
    "age": {
        "65+ years old": "65+",
        "55-64 years old": "55-64",
        "45-54 years old": "45-54",
        "35-44 years old": "35-44",
        "18-24 years old": "18-24",
        "25-34 years old": "25-34",
    },
    "ethnicity_simplified": {
        "Other": "Other",
        "Mixed": "Mixed",
        "Asian": "Asian",
        "Hispanic": "Hispanic",
        "Black": "Black",
        "White": "White",
    },
    "religion_simplified": {
        "Other": "Other",
        "Muslim": "Muslim",
        "Jewish": "Jewish",
        "Christian": "Christian",
        "No Affiliation": "No Affiliation",
    },
    "location_special_region": {
        "Middle East": "Middle East",
        "Northern America": "N. America",
        "Asia": "Asia",
        "Africa": "Africa",
        "Australia and New Zealand": "Aus & NZ",
        "Latin America and the Caribbean": "Latam",
        "UK": "UK",
        "Europe": "Europe",
        "US": "US",
    },
}

##############################################################################
# SET UP COLOR PALETTES
##############################################################################

# From https://www.fabiocrameri.ch/ws/media-library/ce2eb6eee7c345f999e61c02e2733962/readme_scientificcolourmaps.pdf
palettes_dict = {
    "rainbow_trunc": truncate_colormap(cm.roma, 0.1, 0.9),
    "rainbow": cm.roma,
    "roma": cm.roma,
    "romaO": cm.romaO,
    "oslo": cm.oslo,
    "hawaii": cm.hawaii,
    "lipari": cm.lipari,
    "lajolla": cm.lajolla,
    "imola": cm.imola,
    "acton": cm.acton,
    "lapaz": cm.lapaz,
    "bam": cm.bam,
    "bamO": cm.bamO,
    "cork": cm.cork,
    "corkO": cm.corkO,
    "berlin": cm.berlin,
    "managua": cm.managua,
    "vik": cm.vik,
    "vikO": cm.vikO,
    "lisbon": cm.lisbon,
    "tofino": cm.tofino,
    "oleron": cm.oleron,
    "batlow": cm.batlow,
}

# For hard to distinguish categorical variables (with the S)
colorlist_dict = {
    "imola": cm.imolaS.colors,
    "acton": cm.actonS.colors,
    "lajolla": cm.lajollaS.colors,
    "lipari": cm.lipariS.colors,
    "oslo": cm.osloS.colors,
    "batlow": cm.batlowS.colors,
    # Get matplotlib tab 10 too
    "tab10": plt.colormaps["tab10"].colors,
}

spaced_colors = get_equally_spaced_colors(cm.roma, 6, start=0.1, end=0.9)
basic_colors = basic_colors_dict.values()


##############################################################################
# SET UP PLOTTING PARAMETERS
##############################################################################

seaborn_params = {
    "style": "white",
    "palette": spaced_colors,
    "context": "talk",
    "color_codes": True,
}

matplotlib_params = {
    "figure.figsize": (12, 4),
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "axes.prop_cycle": plt.cycler(color=basic_colors),
}

def plot_knowledge_drift():
    # Read CSV file
    csv_path = '/users/grad/abhishektyagi/wanda/wanda/results/plots/knowledge_drift/knowledge_drift_data.csv'
    df = pd.read_csv(csv_path)
    
    # Extract middle points from token ranges
    x_values = []
    for token_range in df['Token']:
        start, end = token_range.split('-')
        middle = (int(start) + int(end)) / 2
        x_values.append(middle)
    
    # Get colors from managua colormap for lines
    line_colors = get_equally_spaced_colors(cm.managua, 2, start=0.2, end=0.8)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_values, df['Unpruned'], marker='o', label='Unpruned', linewidth=2, color=line_colors[0])
    ax.plot(x_values, df['Pruned'], marker='s', label='Pruned', linewidth=2, color=line_colors[1])
    
    # Lollipop points for Unpruned
    # 271-280 green lollipop
    if '271-280' in df['Token'].values:
        y_val = df[df['Token'] == '271-280']['Unpruned'].values[0]
        ax.vlines(x=275.5, ymin=0, ymax=y_val, color=basic_colors_dict['green'], alpha=0.7, linewidth=3)
        ax.scatter(x=275.5, y=y_val, s=150, color=basic_colors_dict['green'], alpha=0.7, zorder=5)
    
    # 201-210 yellow lollipop
    if '201-210' in df['Token'].values:
        y_val = df[df['Token'] == '201-210']['Unpruned'].values[0]
        ax.vlines(x=205.5, ymin=0, ymax=y_val, color=basic_colors_dict['yellow'], alpha=0.7, linewidth=3)
        ax.scatter(x=205.5, y=y_val, s=150, color=basic_colors_dict['yellow'], alpha=0.7, zorder=5, edgecolors='black', linewidths=1)
    
    # Lollipop points for Pruned
    # 501-510 darkgrey lollipop
    if '501-510' in df['Token'].values:
        y_val = df[df['Token'] == '501-510']['Pruned'].values[0]
        ax.vlines(x=505.5, ymin=0, ymax=y_val, color=basic_colors_dict['darkgrey'], alpha=0.7, linewidth=3)
        ax.scatter(x=505.5, y=y_val, s=150, color=basic_colors_dict['darkgrey'], alpha=0.7, zorder=5)
    
    # 431-440 red lollipop
    if '431-440' in df['Token'].values:
        y_val = df[df['Token'] == '431-440']['Pruned'].values[0]
        ax.vlines(x=435.5, ymin=0, ymax=y_val, color=basic_colors_dict['red'], alpha=0.7, linewidth=3)
        ax.scatter(x=435.5, y=y_val, s=150, color=basic_colors_dict['red'], alpha=0.7, zorder=5)
    
    # 271-280 red lollipop
    if '271-280' in df['Token'].values:
        y_val = df[df['Token'] == '271-280']['Pruned'].values[0]
        ax.vlines(x=275.5, ymin=0, ymax=y_val, color=basic_colors_dict['red'], alpha=0.7, linewidth=3)
        ax.scatter(x=275.5, y=y_val, s=150, color=basic_colors_dict['red'], alpha=0.7, zorder=5)
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Knowledge Drift Across Token Positions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show clean intervals at every 50 (0, 50, 100, etc.)
    max_token = max(x_values)
    x_ticks = list(range(0, int(max_token) + 50, 50))
    ax.set_xticks(x_ticks)
    
    # Set x-axis to start at 0 (no gap from origin)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    
    # Save the figure in the same directory as the CSV
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, 'knowledge_drift_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {output_path}")
    plt.close()

def plot_knowledge_drift_time():
    # Read CSV file
    csv_path = '/users/grad/abhishektyagi/wanda/wanda/results/plots/knowledge_drift/knowledge_drift_data.csv'
    df = pd.read_csv(csv_path)
    
    # Extract middle points from token ranges
    x_values = []
    for token_range in df['Token']:
        start, end = token_range.split('-')
        middle = (int(start) + int(end)) / 2
        x_values.append(middle)
    
    df['x_position'] = x_values

    # Apply plotting parameters
    plt.rcParams.update(matplotlib_params)
    
    # Get colors from berlin colormap for lines
    line_colors = get_equally_spaced_colors(cm.berlin, 2, start=0.1, end=0.9)
    
    # Create the plot with the specified style
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot('x_position', 'Unpruned', data=df, color=line_colors[0], linewidth=3, label='Dense', alpha=0.9)
    ax.plot('x_position', 'Pruned', data=df, color=line_colors[1], linewidth=3, label='Sparse', alpha=0.9)
    
    # Lollipop points for Unpruned
    # 271-280 blue lollipop
    if '271-280' in df['Token'].values:
        y_val = df[df['Token'] == '271-280']['Unpruned'].values[0]
        ax.scatter(x=275.5, y=y_val, s=200, color=basic_colors_dict['blue'], alpha=0.8, zorder=5, edgecolors='black', linewidths=1.5)
    
    # 181-190 yellow lollipop with vertical line
    if '181-190' in df['Token'].values:
        y_val = df[df['Token'] == '181-190']['Unpruned'].values[0]
        ax.vlines(x=185.5, ymin=0, ymax=y_val, color=basic_colors_dict['yellow'], alpha=0.5, linewidth=2, linestyle='--')
        ax.scatter(x=185.5, y=y_val, s=150, color=basic_colors_dict['yellow'], alpha=0.8, zorder=5, edgecolors='black', linewidths=1.5)
    
    # Lollipop points for Pruned
    # 501-510 darkgrey lollipop
    if '501-510' in df['Token'].values:
        y_val = df[df['Token'] == '501-510']['Pruned'].values[0]
        ax.vlines(x=505.5, ymin=0, ymax=y_val, color=basic_colors_dict['darkgrey'], alpha=0.5, linewidth=2, linestyle='--')
        ax.scatter(x=505.5, y=y_val, s=150, color=basic_colors_dict['darkgrey'], alpha=0.8, zorder=5, edgecolors='black', linewidths=1.5)
    
    # 431-440 red lollipop
    if '431-440' in df['Token'].values:
        y_val = df[df['Token'] == '431-440']['Pruned'].values[0]
        ax.scatter(x=435.5, y=y_val, s=200, color=basic_colors_dict['red'], alpha=0.8, zorder=5, edgecolors='black', linewidths=1.5)
    
    # 271-280 red lollipop
    if '271-280' in df['Token'].values:
        y_val = df[df['Token'] == '271-280']['Pruned'].values[0]
        ax.scatter(x=275.5, y=y_val, s=200, color=basic_colors_dict['red'], alpha=0.8, zorder=5, edgecolors='black', linewidths=1.5)
    
    # Decoration
    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Similarity Score', fontsize=12, fontweight='bold')
    
    # Set x-axis ticks to show clean intervals at every 50 (0, 50, 100, etc.)
    max_token = max(x_values)
    x_ticks = list(range(0, int(max_token) + 50, 50))
    ax.set_xticks(ticks=x_ticks)
    ax.tick_params(axis='both', labelsize=10)
    
    # Set x-axis to start at 0 (no gap from origin)
    ax.set_xlim(left=0)
    
    # Set y-axis to start at 0 (clamp to origin)
    ax.set_ylim(bottom=0)
    
    # Grid styling
    ax.grid(axis='y', alpha=0.6, linestyle='--')
    
    # Legend styling
    ax.legend(loc='lower left', fontsize=14, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the figure in the same directory as the CSV
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, 'knowledge_drift_plot_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_knowledge_drift()
    plot_knowledge_drift_time()