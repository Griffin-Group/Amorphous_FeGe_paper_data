# Structures

This folder contains the 71 structure snapshots used in Fig. 1, together with the corresponding magnetic-moment files and downstream analysis outputs.

## Contents

- `POSCAR_*`
  - VASP-format structure files for the Fe-Ge configurations studied in the paper.
- `magmom_*.txt`
  - Matching magnetic-moment tables for each structure.
- `mean_field_analysis.py`
  - Script used for the mean-field-style analysis of the structure set.
- `stat_analysis.py`
  - Script used for the statistical analysis of the structure set.
- `statistical_analysis/`
  - Derived plots and tables from the statistical analysis, including neighbor-pattern statistics and correlation/distribution figures.

## Conventions

- Each `POSCAR_n` file is paired with `magmom_n.txt` using the same numeric index.
- The `statistical_analysis/` directory contains generated outputs and can be recreated from the analysis scripts in this folder.
