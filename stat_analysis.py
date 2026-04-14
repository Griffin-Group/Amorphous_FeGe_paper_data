#!/usr/bin/env python3
"""
Statistical analysis of Fe-Ge structures
Focuses on: correlation, distribution, neighbor patterns, and angle type comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ase.io import read
from ase.neighborlist import NeighborList
import glob
import re
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def read_magnetic_moments(filename):
    """Read magnetic moment data from file."""
    magmoms = {}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            start_idx = 1 if lines[0].strip().startswith(('Mx', '#')) else 0
            for i, line in enumerate(lines[start_idx:], start=0):
                if line.strip() and not line.strip().startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        magmoms[i] = float(parts[2]) # Mz moment
                    elif len(parts) == 1:
                        magmoms[i] = float(parts[0])
    except Exception as e:
        print(f"Error reading magnetic moments from {filename}: {str(e)}")
        return {}
    return magmoms

def analyze_single_structure(poscar_path, magmom_path, structure_num):
    """Analyze a single structure and return neighbor statistics for Fe atoms."""
    atoms = read(poscar_path)
    magmoms = read_magnetic_moments(magmom_path)
    symbols = atoms.get_chemical_symbols()
    fe_indices = [i for i, s in enumerate(symbols) if s == 'Fe']
    ge_indices = [i for i, s in enumerate(symbols) if s == 'Ge']

    cutoff = 3.5
    nl = NeighborList([cutoff/2] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    fe_data = []
    for fe_idx in fe_indices:
        neighbors, _ = nl.get_neighbors(fe_idx)
        fe_neighbors, ge_neighbors = [], []
        fe_distances, ge_distances = [], []

        for n_idx in neighbors:
            dist = atoms.get_distance(fe_idx, n_idx, mic=True)
            if n_idx in fe_indices:
                fe_neighbors.append(n_idx)
                fe_distances.append(dist)
            elif n_idx in ge_indices:
                ge_neighbors.append(n_idx)
                ge_distances.append(dist)

        fe_data.append({
            'structure': structure_num,
            'atom_index': fe_idx,
            'magmom': magmoms.get(fe_idx, 0.0),
            'n_fe_neighbors': len(fe_neighbors),
            'n_ge_neighbors': len(ge_neighbors),
            'n_total_neighbors': len(fe_neighbors) + len(ge_neighbors),
            'avg_fe_distance': np.mean(fe_distances) if fe_distances else np.nan,
            'avg_ge_distance': np.mean(ge_distances) if ge_distances else np.nan
        })
    return fe_data

def process_all_structures():
    """Process all POSCAR_N and magmom_N.txt files."""
    poscar_files = glob.glob("POSCAR_*")
    structure_nums = sorted([int(re.search(r'POSCAR_(\d+)', f).group(1)) for f in poscar_files if re.search(r'POSCAR_(\d+)', f)])

    if not structure_nums:
        print("No POSCAR_N files found!")
        return {}, pd.DataFrame()

    all_structure_data, all_fe_data = {}, []
    for num in structure_nums:
        poscar, magmom = f"POSCAR_{num}", f"magmom_{num}.txt"
        if Path(magmom).exists():
            try:
                fe_data = analyze_single_structure(poscar, magmom, num)
                all_structure_data[num] = fe_data
                all_fe_data.extend(fe_data)
            except Exception as e:
                print(f"Error processing structure {num}: {e}")

    return all_structure_data, pd.DataFrame(all_fe_data)

def correlation_analysis(df_stats):
    """1. Correlation Analysis"""
    print("\n=== CORRELATION ANALYSIS ===")
    os.makedirs("statistical_analysis", exist_ok=True)

    # Set publication-quality parameters from get_stat.py
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 12,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.labelpad': 11,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': True,
        'axes.spines.right': True
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    # Colors from get_stat.py
    colors = {
        'fe_neighbors': '#ff7f0e',
        'ge_neighbors': '#2ca02c',
        'total': '#d62728',
        'fe_dist': "#0d7c84",
        'ge_dist': '#8c564b',
        'heatmap': 'coolwarm'
    }

    cols = [('n_fe_neighbors', 'Number of Fe Neighbors', colors['fe_neighbors']),
            ('n_ge_neighbors', 'Number of Ge Neighbors', colors['ge_neighbors']),
            ('n_total_neighbors', 'Total Number of Neighbors', colors['total']),
            ('avg_fe_distance', 'Average Fe–Fe Distance (Å)', colors['fe_dist']),
            ('avg_ge_distance', 'Average Fe–Ge Distance (Å)', colors['ge_dist'])]

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for i, (col, label, color) in enumerate(cols):
        df_plot = df_stats.dropna(subset=[col])
        axes[i].scatter(df_plot[col], df_plot['magmom'], alpha=0.6, color=color, s=26, edgecolors='none', linewidth=0.3)
        axes[i].set_xlabel(label, fontsize=16)
        axes[i].set_ylabel('Magnetic Moment (μ$_B$)', fontsize=16)
        axes[i].text(0.05, 0.95, panel_labels[i], transform=axes[i].transAxes, fontsize=16, fontweight='bold', va='top')

        r, p = stats.pearsonr(df_plot[col], df_plot['magmom'])
        axes[i].text(0.95, 0.05, f'r = {r:.3f}\np = {p:.3f}', transform=axes[i].transAxes,
                     fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.5))

    # Heatmap
    ax = axes[5]
    corr_matrix = df_stats[['magmom', 'n_fe_neighbors', 'n_ge_neighbors', 'n_total_neighbors']].corr()
    im = ax.imshow(corr_matrix, cmap=colors['heatmap'], vmin=-1, vmax=1, aspect='auto')
    for row in range(len(corr_matrix)):
        for col in range(len(corr_matrix.columns)):
            ax.text(col, row, f'{corr_matrix.iloc[row, col]:.2f}', ha="center", va="center", color="black", fontsize=10)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_xticklabels(['Magnet Moment', 'Fe Neighbors', 'Ge Neighbors', 'Total Neighbors'], rotation=45, ha='right')
    ax.set_yticklabels(['Magnet Moment', 'Fe Neighbors', 'Ge Neighbors', 'Total Neighbors'])
    ax.text(0.05, 0.95, panel_labels[5], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', color='white')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Pearson Correlation', fontsize=14)

    plt.tight_layout()
    plt.savefig('statistical_analysis/correlation_analysis.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('statistical_analysis/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def distribution_analysis(df_stats):
    """2. Distribution Analysis"""
    print("\n=== DISTRIBUTION ANALYSIS ===")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 12,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.labelpad': 11,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': True,
        'axes.spines.right': True
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    colors = {
        'magmom': '#1f77b4',
        'fe': '#ff7f0e',
        'ge': '#2ca02c',
        'total': '#d62728',
        'fe_dist': "#0d7c84",
        'ge_dist': '#8c564b',
        'ref_line': "#570FB5"
    }

    dist_cols = [('magmom', 'Magnetic Moment (μ$_B$)', colors['magmom'], 1.183, 'Crystalline'),
                 ('n_fe_neighbors', 'Number of Fe Neighbors', colors['fe'], 6, 'Crystalline'),
                 ('n_ge_neighbors', 'Number of Ge Neighbors', colors['ge'], 7, 'Crystalline'),
                 ('n_total_neighbors', 'Total Number of Neighbors', colors['total'], 13, 'Crystalline'),
                 ('avg_fe_distance', 'Average Fe–Fe Distance (Å)', colors['fe_dist'], 2.8, 'Crystalline'),
                 ('avg_ge_distance', 'Average Fe–Ge Distance (Å)', colors['ge_dist'], 2.35, 'Cryst. 1NN')]

    for i, (col, label, color, ref, ref_label) in enumerate(dist_cols):
        data = df_stats[col].dropna()
        if col in ['magmom', 'avg_fe_distance', 'avg_ge_distance']:
            axes[i].hist(data, bins=30 if col == 'magmom' else 100, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            if col != 'magmom': axes[i].set_xlim(2.2, 3.6)
        else:
            u, c = np.unique(data, return_counts=True)
            axes[i].bar(u, c, width=0.8, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[i].set_xticks(u)

        axes[i].axvline(ref, color=colors['ref_line'], linestyle='--', linewidth=1.5, label=ref_label)
        if col == 'avg_ge_distance':
            axes[i].axvline(2.6, color=colors['ref_line'], linestyle=':', linewidth=1.5, label='Cryst. 2NN')

        axes[i].set_xlabel(label, fontsize=16)
        axes[i].set_ylabel('Count', fontsize=16)
        axes[i].legend(fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig('statistical_analysis/distribution_analysis.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('statistical_analysis/distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def neighbor_pattern_analysis(df_stats):
    """3. Neighbor Pattern Analysis"""
    print("\n=== NEIGHBOR PATTERN ANALYSIS ===")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 12,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.labelpad': 11,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': True,
        'axes.spines.right': True
    })

    df_stats['neighbor_pattern'] = df_stats['n_fe_neighbors'].astype(str) + 'Fe+' + df_stats['n_ge_neighbors'].astype(str) + 'Ge'

    # Analyze magnetic moments by pattern
    pattern_stats = df_stats.groupby('neighbor_pattern')['magmom'].agg(['mean', 'std', 'count']).sort_values('count', ascending=False)

    # Function to get opposite pattern (swap Fe and Ge)
    def get_opposite(p):
        m = re.match(r'(\d+)Fe\+(\d+)Ge', p)
        return f"{m.group(2)}Fe+{m.group(1)}Ge" if m else None

    # Arrange top patterns and their opposites
    arranged = []
    seen = set()
    for p in pattern_stats.index:
        if p not in seen:
            arranged.append(p)
            seen.add(p)
            opp = get_opposite(p)
            if opp in pattern_stats.index and opp not in seen:
                arranged.append(opp)
                seen.add(opp)
        if len(arranged) >= 20: break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top patterns count
    ax = axes[0]
    top_counts = pattern_stats['count'].head(10)
    ax.bar(range(len(top_counts)), top_counts.values, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(top_counts)))
    ax.set_xticklabels(top_counts.index, rotation=45, ha='right')
    ax.set_xlabel('Neighbor Pattern', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    # Magmom by pattern (with opposite pairing logic)
    ax = axes[1]
    plot_labels = arranged[:12]
    plot_means = [pattern_stats.loc[p, 'mean'] for p in plot_labels]

    # Color logic for opposites
    plot_colors = []
    for i, p in enumerate(plot_labels):
        if i > 0 and get_opposite(plot_labels[i-1]) == p:
            plot_colors.append('#ff7f0e') # opposite
        elif i < len(plot_labels)-1 and get_opposite(p) == plot_labels[i+1]:
            plot_colors.append('#1f77b4') # primary
        else:
            plot_colors.append('#2ca02c') # no opposite

    ax.bar(range(len(plot_labels)), plot_means, color=plot_colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add separators between opposite pairs
    for i in range(1, len(plot_labels), 2):
        if i < len(plot_labels) - 1:
            ax.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, rotation=45, ha='right')
    ax.set_xlabel('Neighbor Pattern', fontsize=16)
    ax.set_ylabel('Average Magnetic Moment (μ$_B$)', fontsize=16)
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig('statistical_analysis/neighbor_pattern_analysis.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('statistical_analysis/neighbor_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    pattern_stats.to_csv('statistical_analysis/pattern_statistics.csv')

def analyze_bond_angles(atoms, df_stats, cutoff=3.5, min_distance=2.25):
    """Analyze bond angles Fe_j - Fe_i - Fe_k."""
    fe_indices = [i for i, s in enumerate(atoms.symbols) if s == 'Fe']
    nl = NeighborList([cutoff/2] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    angle_data = []
    for fe_idx in fe_indices:
        magmom = df_stats[df_stats['atom_index'] == fe_idx]['magmom'].iloc[0] if not df_stats[df_stats['atom_index'] == fe_idx].empty else 0.0
        indices, offsets = nl.get_neighbors(fe_idx)

        neighbors = []
        for j, offset in zip(indices, offsets):
            pos_j = atoms.positions[j] + np.dot(offset, atoms.get_cell())
            dist = np.linalg.norm(atoms.positions[fe_idx] - pos_j)
            if min_distance <= dist <= cutoff:
                neighbors.append({'pos': pos_j, 'type': atoms.symbols[j]})

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                vec1, vec2 = neighbors[i]['pos'] - atoms.positions[fe_idx], neighbors[j]['pos'] - atoms.positions[fe_idx]
                cos_angle = np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi

                types = sorted([neighbors[i]['type'], neighbors[j]['type']])
                angle_type = f"{types[0]}-Fe-{types[1]}"
                angle_data.append({'atom_index': fe_idx, 'magmom': magmom, 'angle': angle, 'angle_type': angle_type})

    return pd.DataFrame(angle_data)

def create_combined_angle_type_comparison(combined_df):
    """Create a grid comparing angle types."""

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 12,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.labelpad': 11,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': True,
        'axes.spines.right': True
    })

    angle_types = ['Fe-Fe-Fe', 'Fe-Fe-Ge', 'Ge-Fe-Ge']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {
        'pos': '#d62728',
        'neg': '#1f77b4',
        'zero': '#2ca02c',
        'ref': '#ff7f0e'
    }

    for i, at in enumerate(angle_types):
        data = combined_df[combined_df['angle_type'] == at]
        if data.empty: continue

        # Row 1: Distribution by magmom sign
        ax = axes[0, i]
        pos_data = data[data['magmom'] > 0]
        neg_data = data[data['magmom'] < 0]
        zero_data = data[data['magmom'] == 0]

        bins = np.linspace(data['angle'].min(), data['angle'].max(), 20)
        if not pos_data.empty: ax.hist(pos_data['angle'], bins=bins, alpha=0.7, density=True, color=colors['pos'], label='μ > 0', edgecolor='black', lw=0.3)
        if not neg_data.empty: ax.hist(neg_data['angle'], bins=bins, alpha=0.7, density=True, color=colors['neg'], label='μ < 0', edgecolor='black', lw=0.3)
        if not zero_data.empty: ax.hist(zero_data['angle'], bins=bins, alpha=0.7, density=True, color=colors['zero'], label='μ = 0', edgecolor='black', lw=0.3)

        ax.set_title(f"{at} Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel('Bond Angle (degrees)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 2: Scatter
        ax = axes[1, i]
        ax.scatter(data['angle'], data['magmom'], alpha=0.5, s=20, c=np.where(data['magmom']>0, colors['pos'], np.where(data['magmom']<0, colors['neg'], colors['zero'])), edgecolor='black', lw=0.1)
        ax.axhline(0, color='#570FB5', linestyle='--', lw=1.5)

        if len(data) > 3:
            corr = data['angle'].corr(data['magmom'])
            ax.text(0.05, 0.85, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            z = np.polyfit(data['angle'], data['magmom'], 1)
            ax.plot(data['angle'], np.poly1d(z)(data['angle']), color='red', ls='-', lw=2)

        ax.set_title(f"{at}: Angle vs Magmom", fontsize=14, fontweight='bold')
        ax.set_xlabel('Bond Angle (degrees)', fontsize=12)
        ax.set_ylabel('Magnetic Moment (μ$_B$)', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('statistical_analysis/angle_type_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('statistical_analysis/angle_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Processing structures...")
    all_structs, df_stats = process_all_structures()
    if df_stats.empty: return

    correlation_analysis(df_stats)
    distribution_analysis(df_stats)
    neighbor_pattern_analysis(df_stats)

    print("Analyzing bond angles...")
    os.makedirs("statistical_analysis/bond_angle_analysis", exist_ok=True)
    all_angles = []
    for num in sorted(all_structs.keys()):
        try:
            atoms = read(f"POSCAR_{num}")
            df_struct = df_stats[df_stats['structure'] == num]
            angle_df = analyze_bond_angles(atoms, df_struct)
            if not angle_df.empty:
                angle_df['structure'] = num
                all_angles.append(angle_df)
        except Exception as e:
            print(f"Error in structure {num} angles: {e}")

    if all_angles:
        combined_angles = pd.concat(all_angles, ignore_index=True)
        create_combined_angle_type_comparison(combined_angles)
        combined_angles.to_csv('statistical_analysis/bond_angle_analysis/all_angles.csv', index=False)

    print("\nAnalysis complete. Results in 'statistical_analysis/'")

if __name__ == "__main__":
    main()
