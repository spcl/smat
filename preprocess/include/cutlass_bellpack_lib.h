#pragma once

#define CUTLASS
#ifdef CUTLASS

#include "definitions.h"

// CUDA runtime
#include <cuda_runtime.h>
#include "helper_cuda.h"

template<typename iT, typename T>
int compute_cutlass_bellpack (int rows, int cols, int ell_blocksize, int ellValue_cols, iT* ellColInd, T *ellValues, int B_rows, int B_cols, T *B_vals, int C_rows, int C_cols, T *C_vals, float& dt);

void bellpack_cutlass_multiplyAB(VBR* A, DataT* B, int B_cols, DataT_C* C, int C_cols, float& dt, int verbose);

int cutlass_dense_multiplyAB(int m, int k, DataT* inputA, int n, DataT* inputB, float alp, float bet, DataT_C* output, float& dt);

void cutlas_fixed_blocks_multiply(const VBR& vbmatA, DataT* B, int B_cols, DataT_C* C, float& dt);

void cutlas_blockmat_multiplyBA(const VBR& vbmatA, DataT* B, int B_rows, DataT_C* C, float& dt);

void cutlas_blockmat_multiplyBA_streams(const VBR& vbmatA, DataT* B, int B_rows, DataT_C* C, float& dt, int n_streams);

void cutlas_blockmat_batched(const VBR& vbmatA, DataT* B, int B_cols, DataT_C* C, float& dt);

#endif
