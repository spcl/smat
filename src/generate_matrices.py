import numpy as np
import scipy as sp
import os

def generate_band_mtx_matrix(N: int, b: int, foldername: str = None, filename: str = None):
    """
    Generate a sparse banded matrix of size N x N with bandwidth b.
    Store it to the disc in .mtx format. 
    The values of the matrix are all 1.
    """

    # .mtx format requires edge list. Generate the edge list for band matrix
    edges = []
    for i in range(N):
        for j in range(max(0, i-b), min(N, i+b+1)):
            edges.append((i+1, j+1, 1))

    # .mtx header:
    mtx_header = f'%%MatrixMarket matrix coordinate real general\n% Generated by generate_band_mtx_matrix\n{N} {N} {len(edges)}'

    # Save the .mtx file
    if foldername is None:
        foldername = 'matrices/band_matrices_4_times'

    if filename is None:
        filename = f'band_mtx_{N}_{b}.mtx'

    if not os.path.exists(foldername):
    # Create the folder recursively
    	os.makedirs(foldername)
    	
    with open(os.path.join(foldername, filename), 'w') as f:
        f.write(mtx_header + '\n')
        for edge in edges:
            f.write(' '.join(str(x) for x in edge) + '\n')

import sys

def main():
    n = len(sys.argv)
    if n < 5:
        N_min = 16384
        N_max = 16384
        b_min = 64
        b_max = 16385
    else:
        N_min = sys.argv[1]
        N_max = sys.argv[2]
        b_min = sys.argv[3]
        b_max = sys.argv[4]

    N = N_min
    while (N <= N_max):
        b = b_min
        while (b <= b_max):
            generate_band_mtx_matrix(N, b)
            b *= 4
        N *= 2

if __name__ == '__main__':
    main()