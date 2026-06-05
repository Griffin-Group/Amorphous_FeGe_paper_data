# Monte Carlo Results for Amorphous FeGe

This folder contains Monte Carlo output for amorphous FeGe spin configurations, along with a helper script to compute the topological charge `Q` from `.xsf` snapshots and a table of magnetization and specific heat versus temperature.

## Contents

- `16x16x1/` and `4x4x4/`
  - Temperature-labeled `.xsf` spin snapshots.
  - Each file name encodes the temperature in Kelvin, for example `100.4264022K_spins.xsf`.
  - The `.xsf` files store:
    - the simulation lattice vectors under `PRIMVEC`
    - the spin-site coordinates and spin vectors under `PRIMCOORD`
- `Calculate_Q.py`
  - Python script that reads one `.xsf` spin configuration and computes the per-site topological charge `Q`.
  - It uses a KD-tree to find neighboring spins within a cutoff distance and sums the solid-angle contribution from spin triangles.
  - Output is written to `Q.dat`.
- `M_and_C.dat`
  - Plain-text table of thermodynamic observables versus temperature.
  - Columns:
    - `Temperature (K)`
    - `Magnetization`
    - `C_v`

## `.xsf` snapshot files

The snapshot files are standard XSF-format spin configurations. A typical file contains:

```text
CRYSTAL
PRIMVEC
  a1x a1y a1z
  a2x a2y a2z
  a3x a3y a3z
PRIMCOORD
  N  1
  index  x  y  z  nx  ny  nz
```

Where:

- `x, y, z` are the spin-site coordinates
- `nx, ny, nz` are the spin components
- `N` is the number of spin sites in the snapshot

These files can be opened in standard structure/spin visualization tools that support XSF such as VESTA.

## `Calculate_Q.py`

`Calculate_Q.py` expects a file named `spins.xsf` in the working directory.

### Dependencies

- Python 3
- `numpy`
- `scipy`

### What it does

1. Reads the lattice vectors and spin coordinates from `spins.xsf`
2. Builds a 3x3x3 periodic supercell
3. Finds neighboring spins within a cutoff distance
4. Computes the solid-angle contribution for each spin triangle
5. Writes the non-zero per-site topological charge values to `Q.dat`

### Usage

If you want to compute `Q` for one snapshot, copy or rename the desired `.xsf` file to `spins.xsf`, then run:

```bash
python Calculate_Q.py
```

The script will create:

- `Q.dat`: columns are `x y z nx ny nz Q`

Only sites with `|Q| > 1e-6` are written to the output file.

## `M_and_C.dat`

`M_and_C.dat` is a temperature sweep table containing 240 data points from approximately `0.01 K` to `500 K`.

Example format:

```text
# Temperature (K)   Magnetization   C_v
0.010000631849     0.416149642627  0.534903952082E+06
...
500.001362263864   0.023736307108  0.720016182001E+00
```

You can load it directly into Python, MATLAB, Origin, or similar tools for plotting.

## Suggested workflow

1. Pick a snapshot from `16x16x1/` or `4x4x4/`
2. Rename or copy it to `spins.xsf`
3. Run `python Calculate_Q.py`
4. Analyze `Q.dat` together with `M_and_C.dat`

## Notes

- The snapshot filenames are temperature tagged, so they can be matched with the thermodynamic table.
- The two snapshot directories are organized by lattice size.
- If you use these data in a publication, please cite the relevant simulation and analysis work associated with this dataset.

