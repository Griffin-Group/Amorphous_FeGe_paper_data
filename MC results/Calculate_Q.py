#!/usr/bin/env python3
import numpy as np
from scipy.spatial import cKDTree

def parse_xsf_spins(filename):
    """
    Read a .xsf file with blocks
      PRIMVEC
        a1x a1y a1z
        a2x a2y a2z
        a3x a3y a3z
      PRIMCOORD
        N   1
        idx  x   y   z   nx  ny  nz
        ...
    Returns (coords, spins, lattice) with
      coords  shape (N,3)
      spins   shape (N,3)
      lattice shape (3,3), rows = a1, a2, a3
    """
    coords = []
    spins  = []
    lattice = np.zeros((3,3), dtype=float)

    with open(filename,'r') as f:
        lines = f.readlines()

    # find PRIMVEC
    for i,line in enumerate(lines):
        if line.strip() == 'PRIMVEC':
            iv = i
            break
    else:
        raise RuntimeError("PRIMVEC block not found in " + filename)

    # next three lines are lattice vectors
    for j in range(3):
        parts = lines[iv+1+j].split()
        lattice[j,0] = float(parts[0])
        lattice[j,1] = float(parts[1])
        lattice[j,2] = float(parts[2])

    # find PRIMCOORD
    for i,line in enumerate(lines):
        if line.strip().startswith('PRIMCOORD'):
            ic = i
            break
    else:
        raise RuntimeError("PRIMCOORD block not found in " + filename)

    header = lines[ic+1].split()
    N = int(header[0])
    # read next N lines
    for k in range(N):
        parts = lines[ic+2 + k].split()
        # ignore parts[0] = atom index
        x, y, z   = map(float, parts[1:4])
        nx,ny,nz  = map(float, parts[4:7])
        coords.append((x,y,z))
        spins.append((nx,ny,nz))

    return np.array(coords), np.array(spins), lattice

def main():
    input_file  = 'spins.xsf'
    output_file = 'Q.dat'
    cutoff_distance = 3.5
    tolerance       = 1e-6

    coords, spins, lattice = parse_xsf_spins(input_file)
    N = len(coords)

    # build 3×3×3 supercell shifts
    shifts = []
    for i in (-1,0,1):
        for j in (-1,0,1):
            for k in (-1,0,1):
                shifts.append(i*lattice[0] + j*lattice[1] + k*lattice[2])
    shifts = np.array(shifts)       # shape (27,3)

    # replicate coords and spins
    # coords_images will be (27*N,3); spins_images (27*N,3)
    coords_images = (coords[np.newaxis,:,:] + shifts[:,np.newaxis,:]).reshape(-1,3)
    spins_images  = np.tile(spins, (27,1))

    # build KD‐Tree on images
    tree = cKDTree(coords_images)

    four_pi = 4.0 * np.pi

    with open(output_file, 'w') as fout:
        for i in range(N):
            r_i = coords[i]
            v1  = spins[i]

            # find all image‐atoms within the cutoff
            idxs = tree.query_ball_point(r_i, cutoff_distance)

            # map back to base‐cell indices, drop the central site itself
            neigh = {idx % N for idx in idxs if (idx % N) != i}
            if len(neigh) < 2:
                continue

            Q = 0.0
            # form only triangles (i,j,k)
            for j,k in combinations(neigh, 2):
                v2 = spins[j]
                v3 = spins[k]

                dot12 = np.dot(v1, v2)
                dot23 = np.dot(v2, v3)
                dot31 = np.dot(v3, v1)

                num = 1.0 + dot12 + dot23 + dot31
                den = np.sqrt(2.0*(1+dot12)*(1+dot23)*(1+dot31))

                # clamp the ratio to [-1,1]
                cos_half = num/den
                cos_half = np.clip(cos_half, -1.0, 1.0)

                Omega = 2.0 * np.arccos(cos_half)
                sign  = np.sign(np.dot(v1, np.cross(v2, v3)))

                Q += sign * Omega

            Q /= four_pi

            # write out if not (numerically) zero
            if abs(Q) > tolerance:
                x,y,z   = r_i
                nx,ny,nz = v1
                fout.write(f"{x:.6f}\t{y:.6f}\t{z:.6f}"
                           f"\t{nx:.6f}\t{ny:.6f}\t{nz:.6f}"
                           f"\t{Q:.6e}\n")

    print("Done. Wrote per-site Q to", output_file)

if __name__ == '__main__':
    from itertools import combinations
    main()
