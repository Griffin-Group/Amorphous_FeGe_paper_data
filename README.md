# Data Repository for Amorphous FeGe Paper

[![DOI](https://zenodo.org/badge/1182869832.svg)](https://doi.org/10.5281/zenodo.19053622)

This repository contains the data, structures, analysis scripts, and generated figures supporting the manuscript "Disorder as a route to topological spin textures in amorphous magnets" by T. Bayaraa, Subhashree Satapathy, Peter Fischer, Frances Hellman, and Sinead M. Griffin.

All information needed to reproduce the calculations in the paper is included in this repository.

## Repository Overview

- `Barcode plots/`
  - Inputs and plotting script used to generate the spiral barcode figures for the Fe neighbor analysis.
  - Contains the reference `POSCAR`, the magnetic-moment table `magmom.txt`, the SIA table `SIA.txt`, the plotting script `spiral_barcodes.py`, and example outputs `S9.png` and `S14.png`.
- `Structures/`
  - 71 structure files and matching magnetic-moment tables used for the Fig. 1 structure set.
  - Also includes structural and statistical analysis scripts plus the derived plots and tables.
- `MC results/`
  - Monte Carlo spin snapshots, thermodynamic data, and the helper script used to compute topological charge from `.xsf` configurations.
- Top-level analysis files
  - `Fig3.py` and `Fig3.png` for the interaction analysis figure.
  - `J_DMI.txt` and `SIA_eig_max.txt` as the source tables consumed by `Fig3.py`.
