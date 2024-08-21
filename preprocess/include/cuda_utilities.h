#pragma once

#include "matrices.h"
#include "definitions.h"

// #define PICO_DEBUG
// #define B_ROW_GROWING
// #define B_COL_GROWING
#define BDG_CKP { printf("BDG_CKP: file %s line %d\n", __FILE__, __LINE__); }

#if defined(USE_NVTX)
#include <nvToolsExt.h>

#if !defined(INITIALIZED_NVTX)
#define INITIALIZED_NVTX
const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);
#endif

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

void cublas_fixed_blocks_multiply(const VBR& vbmatA, DataT* B, int B_cols, DataT_C* C, float& dt, int n_streams = 4);

void cublas_blockmat_multiplyBA(const VBR& vbmatA, DataT* B, int B_rows, DataT_C* C, float& dt, int n_streams = 4);

void cublas_blockmat_batched(const VBR& vbmatA, DataT* B, int B_cols, DataT_C* C, float& dt);

void cublas_blockmat_multiplyAB(const VBR& vbmatA, DataT* B, int B_cols, DataT_C* C, float& dt, int n_streams = 16);

//int cublas_gemm_custom(const DataT* A, unsigned int A_rows, unsigned int A_cols, unsigned int lda, const DataT* B, unsigned int B_cols, unsigned int ldb, DataT_C* C, unsigned int ldc, const DataT_C alpha, const DataT_C beta, float& dt);

void pico_print_SpMMM(const char* Aname, int An, int Am, int Az, int* Arows, int* Acols, DataT* Avals, const char* Bname, int Bn, int Bm, DataT* B, const char* Cname, long int Cn, long int Cm, DataT_C* C);

void pico_print_SpMMM(const char* Aname, VBR* A, const char* Bname, int Bn, int Bm, DataT* B, const char* Cname, long int Cn, long int Cm, DataT_C* C);

void pico_print_SpMMM(const char* Aname, int rows, int cols, int ell_blocksize, int ellValue_cols, int ellColumnsInd_rows, int ellColumnsInd_cols, int num_blocks, intT* ellColumnsInd, DataT_C* ellValues, const char* Bname, int Bn, int Bm, DataT* B, const char* Cname, long int Cn, long int Cm, DataT_C* C);

void pico_print_DnM(const char* Cname, int Cn, int Cm, DataT_C* C);

int cusparse_gemm_custom(int rows, int cols, int nnz, int* csrRowPtr, int* csrColInd, DataT* csrVal, DataT* B, int B_cols, int B_lead_dim, DataT_C* C, int C_lead_dim, const DataT_C alpha, const DataT_C beta, float& dt);

int prepare_cusparse_CSR(CSR& cmat, int **csrRowPtr, int **csrColInd, DataT **csrVal);

void cusparse_blockmat_multiplyAB(CSR& vbmatA, DataT* B, int B_cols, DataT_C* C, int C_cols, float& dt);

int cusparse_gemm_custom_ellpack(int rows, int cols, int A_ell_blocksize, int A_ellValues_cols, int A_ellColInd_cols, int A_ellColInd_rows, int A_num_blocks, intT* A_ellColInd, DataT_C* A_ellValues, DataT* B, int B_cols, int B_lead_dim, DataT_C* C, int C_lead_dim, const DataT_C alpha, const DataT_C beta, float& dt);

int prepare_cusparse_BLOCKEDELLPACK(VBR *A, int *ell_blocksize, int* ellValue_cols, int *ellColInd_rows, int *ellColInd_cols, int *num_blocks, intT** ellColInd, DataT_C** ellValues);

void bellpack_blockmat_multiplyAB(VBR* A, DataT* B, int B_cols, DataT_C* C, int C_cols, float& dt, int verbose=0);

DataT* csr2dn (CSR& A);

void cublas_dense_multiplyAB(int rows, int cols, DataT* A, DataT* B, int B_cols, DataT_C* C, float& dt);
