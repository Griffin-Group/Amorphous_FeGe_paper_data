# Barcode Plots

This folder contains the inputs and plotting script used to generate the spiral barcode plots for the Fe neighbor analysis.

## Files

- `POSCAR`
  - Reference structure used to build the Fe-centered neighbor lists.
- `magmom.txt`
  - Site-resolved magnetic moment table used by the plotting script.
- `SIA.txt`
  - Site-resolved SIA data table used by the plotting script.
- `spiral_barcodes.py`
  - Python script that reads the structure and data tables, builds Fe-centered neighbor lists with ASE, and generates the barcode figures.
- `S9.png`
  - Generated barcode plot for magnetic moments.
- `S14.png`
  - Generated barcode plot for SIA eigenvalues.

## Usage

Run the script from this directory so it can find the input files:

```bash
python spiral_barcodes.py
```

The script regenerates `S9.png` and `S14.png` when the required Python packages are installed.

## Dependencies

- Python 3
- `numpy`
- `matplotlib`
- `ase`
