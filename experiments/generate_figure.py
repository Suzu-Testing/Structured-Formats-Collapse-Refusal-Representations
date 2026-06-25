"""Generate PCA visualization figure.

Creates the central figure showing:
- harmful direct (red circles)
- harmless direct (blue circles)  
- harmful tool_call (red triangles)
- harmless tool_call (blue triangles)
- harmful system (red squares)
- harmless system (blue squares)

Makes the thesis visually obvious in 3 seconds.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')
PAPER_DIR = os.path.join(REPO_ROOT, 'paper')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

CSV_DIR = os.path.join(REPO_ROOT, 'csv')


def main():
    df = pd.read_csv(os.path.join(CSV_DIR, 'exp_visualization.csv'))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    markers = {'direct': 'o', 'json': 'D', 'tool_call': '^', 'system': 's'}
    colors = {'harmful': '#d62728', 'harmless': '#1f77b4'}
    format_labels = {'direct': 'Direct', 'json': 'JSON', 'tool_call': 'Tool call', 'system': 'System prompt'}
    
    for fmt in ['direct', 'json', 'tool_call', 'system']:
        for label in ['harmful', 'harmless']:
            mask = (df['format'] == fmt) & (df['label'] == label)
            subset = df[mask]
            
            ax.scatter(
                subset['pca_x'], subset['pca_y'],
                marker=markers[fmt],
                c=colors[label],
                s=60,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
                label=f'{format_labels[fmt]} ({label})',
                zorder=3,
            )
    
    # Draw ellipses around clusters
    for fmt in ['direct', 'tool_call', 'system']:
        mask_fmt = df['format'] == fmt
        subset = df[mask_fmt]
        cx, cy = subset['pca_x'].mean(), subset['pca_y'].mean()
        sx, sy = subset['pca_x'].std() * 2, subset['pca_y'].std() * 2
        
        ellipse = Ellipse(
            (cx, cy), sx, sy,
            fill=False, 
            linestyle='--', 
            linewidth=1.5,
            edgecolor='gray',
            alpha=0.6,
            zorder=1,
        )
        ax.add_patch(ellipse)
        
        # Label the cluster
        offset_y = sy/2 + 3
        ax.annotate(
            format_labels[fmt],
            (cx, cy + offset_y),
            ha='center', va='bottom',
            fontsize=9, fontstyle='italic', color='gray',
            zorder=4,
        )
    
    # Draw a decision boundary (vertical line through direct cluster)
    direct_mask = df['format'] == 'direct'
    direct_data = df[direct_mask]
    boundary_x = direct_data['pca_x'].mean()
    
    # Add arrow showing format shift
    direct_center = (df[df['format'] == 'direct']['pca_x'].mean(), 
                     df[df['format'] == 'direct']['pca_y'].mean())
    tool_center = (df[df['format'] == 'tool_call']['pca_x'].mean(),
                   df[df['format'] == 'tool_call']['pca_y'].mean())
    
    ax.annotate(
        '', xy=tool_center, xytext=direct_center,
        arrowprops=dict(arrowstyle='->', lw=2, color='#2ca02c', connectionstyle='arc3,rad=0.1'),
        zorder=2,
    )
    ax.annotate(
        'Format shift\n(27x > safety gap)',
        xy=((direct_center[0] + tool_center[0])/2, (direct_center[1] + tool_center[1])/2 + 8),
        ha='center', fontsize=9, color='#2ca02c', fontweight='bold',
        zorder=4,
    )
    
    ax.set_xlabel('PC1 (41.3% variance)', fontsize=11)
    ax.set_ylabel('PC2 (16.2% variance)', fontsize=11)
    ax.set_title('Format-Dependent Safety Encoding:\nPreserved Ordering, Shifted Clusters', fontsize=12, fontweight='bold')
    
    # Legend
    ax.legend(
        loc='lower left', 
        fontsize=7.5,
        ncol=2,
        framealpha=0.9,
        edgecolor='gray',
    )
    
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    png_path = os.path.join(FIGURES_DIR, 'pca_format_encoding.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f'Figure saved to: {png_path}')

    pdf_figures = os.path.join(FIGURES_DIR, 'pca_format_encoding.pdf')
    pdf_paper = os.path.join(PAPER_DIR, 'pca_format_encoding.pdf')
    plt.savefig(pdf_figures, bbox_inches='tight')
    plt.savefig(pdf_paper, bbox_inches='tight')
    print(f'PDF saved to: {pdf_figures}')
    print(f'PDF saved to: {pdf_paper}')
    print('DONE.')


if __name__ == '__main__':
    main()
