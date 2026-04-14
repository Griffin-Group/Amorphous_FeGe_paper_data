#!/usr/bin/env python3
"""
Combined script to create barcode visualizations for Fe neighbor analysis.
Generates separate spiral plots for magnetic moments (magmom.txt) and
SIA eigenvalues (SIA.txt).
Uses labeling style from all_barcodes_sorted_by_neighbor_size.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ase.io import read
from ase.neighborlist import NeighborList
import sys
from pathlib import Path

def read_data_file(filename, is_sia=False):
    """
    Read magnetic moment or SIA data from file.
    """
    data = {}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

            if is_sia:
                # Skip header for SIA
                start_idx = 1
                col_idx = 15
            else:
                start_idx = 0
                col_idx = 2

            data_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]

            for i, line in enumerate(data_lines[start_idx:]):
                parts = line.strip().split()
                if len(parts) > col_idx:
                    data[i] = float(parts[col_idx])
    except Exception as e:
        print(f"Warning: Could not read data from {filename}: {str(e)}")
        return {}

    return data

def analyze_neighbors(atoms, cutoff, data_values, min_distance=2.25):
    """
    Analyze Fe neighbors and return data for three cases: Fe-only, Ge-only, and both
    """
    fe_indices = [i for i, symbol in enumerate(atoms.symbols) if symbol == 'Fe']
    ge_indices = [i for i, symbol in enumerate(atoms.symbols) if symbol == 'Ge']

    if not fe_indices:
        raise ValueError("No Fe atoms found in the structure!")

    cutoffs = [cutoff/2] * len(atoms)
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    fe_neighbor_data = {}

    for i in fe_indices:
        fe_dists, ge_dists, all_dists = [], [], []
        indices, offsets = nl.get_neighbors(i)

        for j, offset in zip(indices, offsets):
            pos_i = atoms.positions[i]
            pos_j = atoms.positions[j] + np.dot(offset, atoms.get_cell())
            dist = np.linalg.norm(pos_i - pos_j)

            if min_distance <= dist <= cutoff:
                if j in fe_indices:
                    fe_dists.append(dist)
                    all_dists.append(dist)
                elif j in ge_indices:
                    ge_dists.append(dist)
                    all_dists.append(dist)

        fe_dists.sort()
        ge_dists.sort()
        all_dists.sort()

        fe_neighbor_data[i] = {
            'fe_only': fe_dists,
            'ge_only': ge_dists,
            'both': all_dists,
            'value': data_values.get(i, 0.0),
            'label': i + 1
        }

    return fe_neighbor_data

def create_barcode_plot_spiral(ax, fe_neighbor_data, case='fe_only',
                              cmap_name='viridis', min_distance=2.25,
                              title="", subplot_label=""):
    """
    Create a spiral barcode plot matching all_barcodes_sorted_by_neighbor_size.py style
    """
    values = [data['value'] for data in fe_neighbor_data.values()]
    vmin, vmax = min(values), max(values)

    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    sorted_fe_indices = sorted(fe_neighbor_data.keys(),
                              key=lambda x: len(fe_neighbor_data[x][case]),
                              reverse=False)

    n_fe = len(sorted_fe_indices)
    theta_positions = np.linspace(0, 2*np.pi, n_fe, endpoint=False)

    r_min, r_max = min_distance, 3.5
    theta_labels = []

    for idx, fe_idx in enumerate(sorted_fe_indices):
        data = fe_neighbor_data[fe_idx]
        distances = data[case]
        val = data['value']
        color = cmap(norm(val))

        theta = theta_positions[idx]
        n_neighbors = len(data[case])
        theta_labels.append(f"{n_neighbors}")

        for dist in distances:
            if min_distance <= dist <= r_max:
                theta_width = 2*np.pi / n_fe * 0.8
                radial_width = 0.02
                theta_range = np.linspace(theta - theta_width/2, theta + theta_width/2, 20)
                r_inner = dist - radial_width/2
                r_outer = dist + radial_width/2
                ax.fill_between(theta_range, r_inner, r_outer, color=color, alpha=0.8)

    ax.set_rlim(r_min, r_max)

    # Matching style from all_barcodes_sorted_by_neighbor_size.py
    ax.set_xticks(theta_positions)
    ax.set_xticklabels(theta_labels, fontsize=10)

    r_ticks = np.arange(2.25, 3.6, 0.5)
    ax.set_rticks(r_ticks)
    ax.set_rlabel_position(90)

    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Add (a), (b), (c) label
    ax.text(-0.1, 1.15, subplot_label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm

def main():
    poscar_file = "POSCAR"
    if not Path(poscar_file).exists():
        print(f"Error: {poscar_file} not found!")
        sys.exit(1)

    atoms = read(poscar_file)
    cutoff = 3.5
    cases = ['fe_only', 'ge_only', 'both']
    case_titles = {
        'fe_only': 'Fe Neighbors',
        'ge_only': 'Ge Neighbors',
        'both': 'All Neighbors'
    }
    subplot_labels = ['(a)', '(b)', '(c)']

    # 1. Process Magnetic Moments
    magmoms = read_data_file("magmom.txt", is_sia=False)
    if magmoms:
        fe_data = analyze_neighbors(atoms, cutoff, magmoms)
        fig = plt.figure(figsize=(20, 6))
        axes = []
        for i, case in enumerate(cases):
            ax = fig.add_subplot(1, 4, i+1, projection='polar')
            sm = create_barcode_plot_spiral(
                ax, fe_data, case=case, cmap_name='viridis',
                title=case_titles[case], subplot_label=subplot_labels[i]
            )
            axes.append(ax)

        cbar = plt.colorbar(sm, ax=axes[-1], label='Magnetic Moment ($μ_B$)', shrink=0.5, pad=0.1)
        plt.tight_layout(pad=2.0)
        plt.savefig("S9.png", dpi=300, bbox_inches='tight')
        print("Saved: S9.png")

    # 2. Process SIA
    sia_values = read_data_file("SIA.txt", is_sia=True)
    if sia_values:
        fe_data_sia = analyze_neighbors(atoms, cutoff, sia_values)
        fig = plt.figure(figsize=(20, 6))
        axes = []
        for i, case in enumerate(cases):
            ax = fig.add_subplot(1, 4, i+1, projection='polar')
            sm = create_barcode_plot_spiral(
                ax, fe_data_sia, case=case, cmap_name='coolwarm',
                title=case_titles[case], subplot_label=subplot_labels[i]
            )
            axes.append(ax)

        cbar = plt.colorbar(sm, ax=axes[-1], label='Max absolute eigenvalue of SIA (meV)', shrink=0.5, pad=0.1)
        plt.tight_layout(pad=2.0)
        plt.savefig("S14.png", dpi=300, bbox_inches='tight')
        print("Saved: S14.png")

    plt.show()

if __name__ == "__main__":
    main()
