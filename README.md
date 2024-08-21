# SMaT: (S)parse (Ma)trix Matrix (T)ensor Core-accelerated library ([PDF](link))

## Abstract

High-performance sparse matrix–matrix (SpMM)
multiplication is paramount for science and industry, as the ever-
increasing sizes of data prohibit using dense data structures.
Yet, existing hardware, such as Tensor Cores (TC), is ill-suited for SpMM, as it imposes strict constraints on data structures
that cannot be met by unstructured sparsity found in many
applications. To address this, we introduce (S)parse (Ma)trix
Matrix (T)ensor Core-accelerated (SMaT): a novel SpMM library
that utilizes TCs for unstructured sparse matrices. Our block-
sparse library leverages the low-level CUDA MMA (matrix-
matrix-accumulate) API, maximizing the performance offered by
modern GPUs. Algorithmic optimizations such as sparse matrix
permutation, further improve performance by minimizing the
number of non-zero blocks. The evaluation on NVIDIA A100
shows that SMaT outperforms SotA libraries (DASP, cuSPARSE,
and Magicube) by up to 125x (on average 2.6x). SMaT can be
used to accelerate many workloads in scientific computing, large
model training, inference, and others.

## Requirements

### Hardware 
We run our experiments on the Swiss National Computing Center’s Ault compute cluster. Each node
is equipped with a single NVIDIA A100-SXM4-40GB GPU,
and AMD EPYC 7742 @ 2.25GHz CPU. The A100 driver
version is 530.30.02.

### Software 
All experiments were executed using the GCC
12.3.0 compiler, NVIDIA nvcc v12.0, NVIDIA cuSPARSE
v12.0, NVIDIA CUDA Toolkit v12.0, Python 3.9, and the
following Python libraries: Pandas, Matplotlib, Numpy, Scipy,
and Seaborn


To create the environment:
```bash
conda env create -f smat_env.yml
conda activate smat
sudo apt-get install libgflags-dev
```

## Datasets
For preparing the matrices run the following:

- SuiteSparse Collection:
```bash
python download_suitesparse.py
```
- Synthetic band matrices:
```bash
python generate_matrices.py
```


## Compiling
In order to compile the library:
```bash
cd src/cuda_hgemm
source compile.sh
```

## Running The Code
Point `<path>` inside `src/run_smat.sh` to input matrix locations, and run:
```bash
cd src/
source run_smat.sh
```