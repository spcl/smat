
import ssgetpy
import numpy as np
import glob, os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Plots for multiplication experiments')
parser.add_argument('-n',
                    '--nmin',
                    type=int,
                    default=20000,
                    help='The minimum row and column size for matrices to download')
parser.add_argument('-N',
                    '--nmax',
                    type=int,
                    default=100000,
                    help='The maximum row and column size for matrices to download')
parser.add_argument('-d',
                    '--densitymin',
                    type = float,
                    default=0.0001,
                    help='The minimum density for matrices to download')
parser.add_argument('-D',
                    '--densitymax',
                    type = float,
                    default=0.1,
                    help='The maximum density for matrices to download')
parser.add_argument('-o',
                    '--outputdir',
                    type = str,
                    default='data/suitsparse_autocollect/',
                    help='The directory where to download the matrices')
args = vars(parser.parse_args())

Ns = (args["nmin"],args["nmax"])
densities = (args["densitymin"],args["densitymax"])
download_path = f"{args['outputdir']}/suitsparse_N{Ns[0]}_{Ns[1]}_dN{densities[0]}_{densities[1]}"


precision = 1000
matrices = []
for n in range(Ns[0],Ns[1],precision):
    rowbounds = (n,n+precision)
    colbounds = (n,n+precision)
    nzbounds = (int(n*n*densities[0]), int(n*n*densities[1]))
    matrices += ssgetpy.search(rowbounds = rowbounds, colbounds = colbounds, nzbounds = nzbounds)


print(f"found {len(matrices)} matrices with {Ns[0]} < N,M < {Ns[1]} and {densities[0]} < density < {densities[1]}")
for matrix in matrices:
    if len(glob.glob(f"{download_path}/*{matrix.name}*")) != 0:
        print("Already downloaded", matrix.name, ", skipping")
    else:

        matrix.download(extract=True, destpath = f"{download_path}")
        files = glob.glob(f"{download_path}/*{matrix.name}*/*.*")
        mainfile = min(files, key=len)
        print("downloaded", matrix.name)
        dir = os.path.dirname(os.path.realpath(mainfile))
        os.rename(mainfile,f"{download_path}/{matrix.name}.mtx")
        shutil.rmtree(dir)	
