#!/usr/bin/env python3
"""
Mean field analysis of Fe magnetic moments.
Generates 'mean_field_analysis.png' based on POSCAR_N and magmom_N.txt files.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import NeighborList
from scipy.stats import pearsonr

def read_magnetic_moments(filepath):
    """Read magnetic moment data from file."""
    magmoms = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            start_idx = 1 if lines and (lines[0].strip().startswith('Mx') or lines[0].strip().startswith('#')) else 0
            for i, line in enumerate(lines[start_idx:]):
                if line.strip() and not line.strip().startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        magmoms[i] = float(parts[2]) # Mz moment
                    elif len(parts) == 1:
                        magmoms[i] = float(parts[0])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}
    return magmoms

def calculate_mean_field(poscar_path, magmom_path, structure_num):
    """Calculate mean field for each Fe atom in a structure."""
    atoms = read(poscar_path)
    magmoms = read_magnetic_moments(magmom_path)
    symbols = atoms.get_chemical_symbols()
    fe_indices = [i for i, symbol in enumerate(symbols) if symbol == 'Fe']

    cutoff = 3.5
    nl = NeighborList([cutoff/2] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    results = []
    for fe_idx in fe_indices:
        neighbors, _ = nl.get_neighbors(fe_idx)
        neighbor_magmoms = []
        fe_neighbor_count = 0
        for n_idx in neighbors:
            dist = atoms.get_distance(fe_idx, n_idx, mic=True)
            if dist >= 2.25 and n_idx in fe_indices:
                neighbor_magmoms.append(magmoms.get(n_idx, 0.0))
                fe_neighbor_count += 1

        results.append({
            'structure': structure_num,
            'atom_index': fe_idx,
            'magmom': magmoms.get(fe_idx, 0.0),
            'mean_field': sum(neighbor_magmoms) if neighbor_magmoms else 0.0,
            'n_fe_neighbors': fe_neighbor_count,
            'avg_neighbor_magmom': np.mean(neighbor_magmoms) if neighbor_magmoms else 0.0
        })
    return results

def main():
    # 1. Collect data
    all_data = []
    poscar_files = glob.glob("POSCAR_*")
    struct_nums = sorted([int(re.search(r'POSCAR_(\d+)', f).group(1)) for f in poscar_files if re.search(r'POSCAR_(\d+)', f)])

    for i in struct_nums:
        poscar, magmom = f'POSCAR_{i}', f'magmom_{i}.txt'
        if os.path.exists(magmom):
            try:
                all_data.extend(calculate_mean_field(poscar, magmom, i))
                print(f"Processed structure {i}")
            except Exception as e:
                print(f"Error in structure {i}: {e}")

    if not all_data:
        print("No data collected!")
        return
    df = pd.DataFrame(all_data)

    # 2. Plotting
    plt.rcParams.update({
        'font.size': 12, 'axes.linewidth': 0.8, 'xtick.major.size': 4, 'ytick.major.size': 4,
        'axes.spines.top': True, 'axes.spines.right': True
    })

    colors = {'zero_line': '#570FB5', 'positive': '#d62728', 'negative': '#1f77b4', 'zero': '#2ca02c', 'regression': '#ff0000'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # (a) Scatter: magmom vs mean_field
    ax = axes[0, 0]
    sc = ax.scatter(df['mean_field'], df['magmom'], c=df['n_fe_neighbors'], cmap='viridis', alpha=0.7, s=50, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Mean Field (μ$_B$)'), ax.set_ylabel('Magnetic Moment (μ$_B$)')
    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    plt.colorbar(sc, ax=ax, shrink=0.8).set_label('Number of Fe Neighbors')
    if len(df) > 2:
        corr, _ = pearsonr(df['mean_field'], df['magmom'])
        ax.text(0.05, 0.85, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.9))
        z = np.polyfit(df['mean_field'], df['magmom'], 1)
        ax.plot(df['mean_field'], np.poly1d(z)(df['mean_field']), color=colors['regression'], ls='--', lw=2)
    ax.axhline(0, color=colors['zero_line'], ls='--', alpha=0.7), ax.axvline(0, color=colors['zero_line'], ls='--', alpha=0.7)

    # (b) 2D Histogram
    ax = axes[0, 1]
    h2d = ax.hist2d(df['mean_field'], df['magmom'], bins=30, cmap='YlOrRd', alpha=0.8)
    ax.set_xlabel('Mean Field (μ$_B$)'), ax.set_ylabel('Magnetic Moment (μ$_B$)')
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.axhline(0, color='white', ls='--', alpha=0.8), ax.axvline(0, color='white', ls='--', alpha=0.8)
    plt.colorbar(h2d[3], ax=ax, shrink=0.8).set_label('Count')

    # (c) Distribution by MF sign
    ax = axes[1, 0]
    bins = np.linspace(df['magmom'].min(), df['magmom'].max(), 25)
    for sign, label, color in [(df['mean_field'] > 0, 'Positive', 'positive'), (df['mean_field'] < 0, 'Negative', 'negative'), (df['mean_field'] == 0, 'Zero', 'zero')]:
        if sign.any():
            ax.hist(df[sign]['magmom'], bins=bins, alpha=0.7, density=True, color=colors[color], edgecolor='black', lw=0.5, label=f'{label} MF (n={sign.sum()})')
    ax.set_xlabel('Magnetic Moment (μ$_B$)'), ax.set_ylabel('Probability Density')
    ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.axvline(0, color=colors['zero_line'], ls='--', alpha=0.7), ax.legend(fontsize=10)

    # (d) Avg neighbor magmom vs own magmom
    ax = axes[1, 1]
    df_n = df[df['n_fe_neighbors'] > 0]
    if not df_n.empty:
        ax.scatter(df_n['avg_neighbor_magmom'], df_n['magmom'], alpha=0.7, s=50, color='#1f77b4', edgecolor='black', linewidth=0.3)
        ax.set_xlabel('Avg Neighbor Magmom (μ$_B$)'), ax.set_ylabel('Magnetic Moment (μ$_B$)')
        ax.text(0.05, 0.95, '(d)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, 'k--', alpha=0.6, label='y = x')
        if len(df_n) > 2:
            c, _ = pearsonr(df_n['avg_neighbor_magmom'], df_n['magmom'])
            ax.text(0.05, 0.85, f'r = {c:.3f}', transform=ax.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.9))
            z = np.polyfit(df_n['avg_neighbor_magmom'], df_n['magmom'], 1)
            ax.plot(df_n['avg_neighbor_magmom'], np.poly1d(z)(df_n['avg_neighbor_magmom']), color=colors['regression'], label='Reg.')
        ax.axhline(0, color=colors['zero_line'], ls='--', alpha=0.7), ax.axvline(0, color=colors['zero_line'], ls='--', alpha=0.7)
        ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('statistical_analysis/mean_field_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved mean_field_analysis.png")

if __name__ == "__main__":
    main()
