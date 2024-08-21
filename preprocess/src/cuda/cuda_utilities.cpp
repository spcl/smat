//CUDA Utilities and system includes
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "driver_types.h"
#include "helper_cuda.h"

//others
#include <stdio.h>
#include <iostream>
#include <typeinfo>

//includes from SPARTA
#include "cuda_utilities.h"
#include "matrices.h"
#include "utilities.h"
#include <numeric>
#include <vector>

#define Z_DATATYPE_AB CUDA_R_8I
#define Z_DATATYPE_C CUDA_R_32I
#define Z_CUBLAS_COMPUTE CUBLAS_COMPUTE_32I
// BUG? Does it needs Z_CUSPARSE_COMPUTE?

#define Q_DATATYPE_AB CUDA_R_16F
#define Q_DATATYPE_C CUDA_R_16F
#define Q_CUBLAS_COMPUTE CUBLAS_COMPUTE_16F
#define Q_CUSPARSE_COMPUTE CUDA_R_16F       // NOTE: cusparse_gemm_custom not supports CUDA_R_16F, fixed as CUDA_R_32F

/*ONLY WORKS WITH 
    DataT = float, double, int;
    intT = int
*/

void cublas_fixed_blocks_multiply(const VBR& vbmatA, DataT* B, int B_cols, DataT_C* C, float& dt, int n_streams)
{
//multiplies a fixed size VBR matrix (vbmatA) and dense matrix (B); stores A*B into (C)
    //vbmatA:       column-major entries (in-block) storage;
    //              row-major block storage; 
    //B:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    //C:            column-major storage; TODO: allow general storage format (implement through cublas transpose)


    cudaDataType_t data_type_AB;
    cudaDataType_t data_type_C;
    cublasComputeType_t compute_type;

    if (typeid(DataT) == typeid(int8_t))
    {
        data_type_AB = Z_DATATYPE_AB;
        data_type_C =  Z_DATATYPE_C;
        compute_type = Z_CUBLAS_COMPUTE;
    }
    else if (typeid(DataT) == typeid(float))
    {
        data_type_AB = Q_DATATYPE_AB;
        data_type_C =  Q_DATATYPE_C;
        compute_type = Q_CUBLAS_COMPUTE;
    }
    else
    {
        std::cout << "WARNING! Unsopported multiplication type in cublas_blockmat_multiply(). Check matrices.h" << std::endl;
    }

    cublasGemmAlgo_t cuda_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    intT A_rows = vbmatA.rows;
    intT A_cols = vbmatA.cols;

    intT B_rows = A_cols;
    intT C_rows = A_rows;
    int C_cols = B_cols;

    const DataT_C alpha = 1;
    const DataT_C beta = 1;

    //allocate memory on device
    intT size_A = vbmatA.nztot; //total nonzero entries in vbmat
    intT mem_size_A = sizeof(DataT) * size_A;

    intT size_B = B_rows * B_cols;
    intT mem_size_B = sizeof(DataT) * size_B;

    intT size_C = C_rows * C_cols;
    intT mem_size_C = sizeof(DataT_C) * size_C;

    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    DataT* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy to device the vbmat matrix (nonzero blocks are stored consecutively and in column major format)
    checkCudaErrors(cublasSetVector(
        size_A, sizeof(DataT), vbmatA.mab, 1, d_A, 1));

    //copy B to device (maybe more efficient to copy it block by block?)
    checkCudaErrors(cudaMemcpy(d_B, B, B_rows * B_cols * sizeof(DataT), cudaMemcpyHostToDevice));
    // ----------------------------------------------------------------------
    
    intT max_blocks_in_row = 0;
    intT tot_nz_blocks = 0;
    intT row_block_size = vbmatA.row_part[1] - vbmatA.row_part[0];
    intT block_area = row_block_size*vbmatA.block_col_size;
    intT* current_jab = vbmatA.jab;
    DataT* current_mab = d_A;
    std::vector<intT*> jab_positions;
    std::vector<DataT*> mab_positions;
    for (intT ib = 0; ib < vbmatA.block_rows; ib++)
        {
            jab_positions.push_back(current_jab);
            mab_positions.push_back(current_mab);
            current_jab += vbmatA.nzcount[ib];
            current_mab += vbmatA.nzcount[ib]*vbmatA.block_col_size*row_block_size;
            max_blocks_in_row = std::max(max_blocks_in_row, vbmatA.nzcount[ib]);
            tot_nz_blocks+= vbmatA.nzcount[ib];
        }


    if (n_streams > vbmatA.block_rows) n_streams = vbmatA.block_rows;
    cudaStream_t streams[n_streams];
    for (intT ib = 0; ib < n_streams; ib++)
    {
        cudaStreamCreateWithFlags(&streams[ib],cudaStreamNonBlocking);
    }

    //initialize cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //creates streams. Each block rows is assigned a different stream.
    DataT* d_C_block;
    intT jb;
    cublasSetStream(handle, streams[0]);               //each stream handles at most max_blocks_per_stream of block_rows    
    //loop through all blocks
    for(intT nzs = 0; nzs < max_blocks_in_row; nzs++ )      //loop horizontally through block rows
    {        
        for (intT ib = 0; ib < vbmatA.block_rows; ib++)
        {
            if (nzs >= vbmatA.nzcount[ib]) continue;
            jb = *(jab_positions[ib] + nzs);
            const DataT* d_B_block = d_B + vbmatA.block_col_size*jb;    //access the vertical block of B that is going to be multiplied with blocks of A in block-row ib
	        const DataT* d_A_block = mab_positions[ib] + nzs*block_area;           //access the block on d_A.
            d_C_block = d_C + vbmatA.row_part[ib];                            //access the block on d_C.

//            int k = vbmatA.block_col_size, m = B_cols, n = row_block_size;
//          //m, n, k <-- block_A: m*k   block_B: k*n   block_C: m*n
    
            cublasSetStream(handle, streams[ib%n_streams]);               //each stream handles at most max_blocks_per_stream of block_rows    
            //multiply the blocks, store result in d_C_block
            checkCudaErrors(
                cublasGemmEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    row_block_size,B_cols,vbmatA.block_col_size,                                               //m, n, k <-- block_A: m*k   block_B: k*n   block_C: m*n
                    &alpha,
                    d_A_block,                                          // blockA device pointer,
                    data_type_AB,                                       // blockA datatype
                    row_block_size,                                      // blockA leading dimension
                    d_B_block,                                          // blockB device pointer
                    data_type_AB,                                       // blockB datatype
                    B_rows,                                             // B leading dimension
                    &beta,
                    d_C_block, data_type_C,                             // blockC device pointer, blockC type
                    C_rows,                                             // C leading dimension
                    compute_type,                                       // compute_type
                    cuda_algo)
            );                                       
            
            //move mab and jab pointers forward
	    }

    }

    //record the elapsed time onto dt
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < n_streams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    //let each stream copy the relevant C block from device
    checkCudaErrors(cublasGetMatrix(
            C_rows, C_cols, sizeof(DataT_C), d_C, C_rows, C, C_rows));

    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    checkCudaErrors(cublasDestroy(handle));
}

void cublas_fixed_blocks_multiply_v2(const VBR& vbmatA, DataT* B, int B_cols, DataT_C* C, float& dt, int n_streams)
{
//multiplies a fixed size VBR matrix (vbmatA) and dense matrix (B); stores A*B into (C)
    //vbmatA:       column-major entries (in-block) storage;
    //              row-major block storage; 
    //B:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    //C:            column-major storage; TODO: allow general storage format (implement through cublas transpose)


    cudaDataType_t data_type_AB;
    cudaDataType_t data_type_C;
    cublasComputeType_t compute_type;

    if (typeid(DataT) == typeid(int8_t))
    {
        data_type_AB = Z_DATATYPE_AB;
        data_type_C =  Z_DATATYPE_C;
        compute_type = Z_CUBLAS_COMPUTE;
    }
    else if (typeid(DataT) == typeid(float))
    {
        data_type_AB = Q_DATATYPE_AB;
        data_type_C =  Q_DATATYPE_C;
        compute_type = Q_CUBLAS_COMPUTE;
    }
    else
    {
        std::cout << "WARNING! Unsopported multiplication type in cublas_blockmat_multiply(). Check matrices.h" << std::endl;
    }

    cublasGemmAlgo_t cuda_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    intT A_rows = vbmatA.rows;
    intT A_cols = vbmatA.cols;

    intT B_rows = A_cols;
    intT C_rows = A_rows;
    int C_cols = B_cols;

    const DataT_C alpha = 1;
    const DataT_C beta = 1;

    //allocate memory on device
    intT size_A = vbmatA.nztot; //total nonzero entries in vbmat
    intT mem_size_A = sizeof(DataT) * size_A;

    intT size_B = B_rows * B_cols;
    intT mem_size_B = sizeof(DataT) * size_B;

    intT size_C = C_rows * C_cols;
    intT mem_size_C = sizeof(DataT_C) * size_C;

    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    DataT* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy to device the vbmat matrix (nonzero blocks are stored consecutively and in column major format)
    checkCudaErrors(cublasSetVector(
        size_A, sizeof(DataT), vbmatA.mab, 1, d_A, 1));

    //copy B to device (maybe more efficient to copy it block by block?)
    checkCudaErrors(cudaMemcpy(d_B, B, B_rows * B_cols * sizeof(DataT), cudaMemcpyHostToDevice));
    // ----------------------------------------------------------------------
    
    intT max_blocks_in_row = 0;
    intT tot_nz_blocks = 0;
    intT row_block_size = vbmatA.row_part[1] - vbmatA.row_part[0];
    intT block_area = row_block_size*vbmatA.block_col_size;
    intT* current_jab = vbmatA.jab;
    DataT* current_mab = d_A;
    std::vector<intT*> jab_positions;
    std::vector<DataT*> mab_positions;
    for (intT ib = 0; ib < vbmatA.block_rows; ib++)
        {
            jab_positions.push_back(current_jab);
            mab_positions.push_back(current_mab);
            current_jab += vbmatA.nzcount[ib];
            current_mab += vbmatA.nzcount[ib]*vbmatA.block_col_size*row_block_size;
            max_blocks_in_row = std::max(max_blocks_in_row, vbmatA.nzcount[ib]);
            tot_nz_blocks+= vbmatA.nzcount[ib];
        }


    if (n_streams > vbmatA.block_rows) n_streams = vbmatA.block_rows;
    cudaStream_t streams[n_streams];
    for (intT ib = 0; ib < n_streams; ib++)
    {
        cudaStreamCreateWithFlags(&streams[ib],cudaStreamNonBlocking);
    }

    //initialize cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //creates streams. Each block rows is assigned a different stream.
    DataT* d_C_block;
    intT jb;
    cublasSetStream(handle, streams[0]);               //each stream handles at most max_blocks_per_stream of block_rows    
    //loop through all blocks 
    for (intT ib = 0; ib < vbmatA.block_rows; ib++)
    {
        d_C_block = d_C + vbmatA.row_part[ib];                            //access the block on d_C.
        cublasSetStream(handle, streams[ib%n_streams]);               //each stream handles at most max_blocks_per_stream of block_rows    
        for(intT nzs = 0; nzs < vbmatA.nzcount[ib]; nzs++ )      //loop horizontally through block rows
        {
            jb = *(jab_positions[ib] + nzs);
            const DataT* d_B_block = d_B + vbmatA.block_col_size*jb;    //access the vertical block of B that is going to be multiplied with blocks of A in block-row ib
	        const DataT* d_A_block = mab_positions[ib] + nzs*block_area;           //access the block on d_A.

//            int k = vbmatA.block_col_size, m = B_cols, n = row_block_size;
//          //m, n, k <-- block_A: m*k   block_B: k*n   block_C: m*n
    
            //multiply the blocks, store result in d_C_block
            checkCudaErrors(
                cublasGemmEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    row_block_size,B_cols,vbmatA.block_col_size,                                               //m, n, k <-- block_A: m*k   block_B: k*n   block_C: m*n
                    &alpha,
                    d_A_block,                                          // blockA device pointer,
                    data_type_AB,                                       // blockA datatype
                    row_block_size,                                      // blockA leading dimension
                    d_B_block,                                          // blockB device pointer
                    data_type_AB,                                       // blockB datatype
                    B_rows,                                             // B leading dimension
                    &beta,
                    d_C_block, data_type_C,                             // blockC device pointer, blockC type
                    C_rows,                                             // C leading dimension
                    compute_type,                                       // compute_type
                    cuda_algo)
            );                                       
            
            //move mab and jab pointers forward
	    }

    }

    //record the elapsed time onto dt
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    //let each stream copy the relevant C block from device
    checkCudaErrors(cublasGetMatrix(
            C_rows, C_cols, sizeof(DataT_C), d_C, C_rows, C, C_rows));

    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    checkCudaErrors(cublasDestroy(handle));
}

void cublas_blockmat_multiplyBA_test(const VBR& vbmatA, DataT* B, int B_rows, DataT_C* C, float& dt, int n_streams)
{
//multiplies a VBS matrix (vbmatA) and dense matrix (B); stores B*A into (C)
    //vbmatA:       column-major entries (in-block) storage;
    //              row-major block storage; 
    //B:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    //C:            column-major storage; TODO: allow general storage format (implement through cublas transpose)


    cudaDataType_t data_type_AB;
    cudaDataType_t data_type_C;
    cublasComputeType_t compute_type;

    if (typeid(DataT) == typeid(int8_t))
    {
        data_type_AB = Z_DATATYPE_AB;
        data_type_C =  Z_DATATYPE_C;
        compute_type = Z_CUBLAS_COMPUTE;
    }
    else if (typeid(DataT) == typeid(float))
    {
        data_type_AB = Q_DATATYPE_AB;
        data_type_C =  Q_DATATYPE_C;
        compute_type = Q_CUBLAS_COMPUTE;
    }
    else
    {
        std::cout << "WARNING! Unsopported multiplication type in cublas_blockmat_multiply(). Check matrices.h" << std::endl;
    }


    cublasGemmAlgo_t cuda_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    intT A_rows = vbmatA.rows;
    intT A_cols = vbmatA.cols;
    intT B_cols = A_rows;
    intT C_rows = B_rows;
    intT C_cols = A_cols;

    const DataT_C alpha = 1;
    const DataT_C beta = 1;

    //allocate memory on device
    intT size_A = vbmatA.nztot; //total nonzero entries in vbmat
    intT mem_size_A = sizeof(DataT) * size_A;

    intT size_B = B_rows * B_cols;
    intT mem_size_B = sizeof(DataT) * size_B;

    intT size_C = C_rows * C_cols;
    intT mem_size_C = sizeof(DataT_C) * size_C;

    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    DataT* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy to device the vbmat matrix (nonzero blocks are stored consecutively and in column major format)
    checkCudaErrors(cublasSetVector(
        size_A, sizeof(DataT), vbmatA.mab, 1, d_A, 1));

    //copy B to device (maybe more efficient to copy it block by block?)
    checkCudaErrors(cudaMemcpy(d_B, B, B_rows * B_cols * sizeof(DataT), cudaMemcpyHostToDevice));
    // ----------------------------------------------------------------------

    //initialize cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //creates streams. Each block rows is assigned a different stream.
    
    if (n_streams > vbmatA.block_cols) n_streams = vbmatA.block_cols;
    cudaStream_t streams[n_streams];
    for (intT ib = 0; ib < n_streams; ib++)
    {
	cudaStreamCreate(&(streams[ib]));
        // cudaStreamCreateWithFlags(&streams[ib],cudaStreamNonBlocking);
    }

    intT mat_idx = 0; //keeps writing position for mat
    intT vbmat_idx = 0; //keeps reading position for vbmat 
    intT ja_count = 0; //keeps total nonzero blocks count;
    intT tot_nonzero_blocks = 0; //index for total nonzero blocks
    intT rows_in_block;
    intT size_block, mem_size_block;
    intT* jab_loc = vbmatA.jab;
    intT stream_col_assign[vbmatA.block_cols]{-1};
    int current_stream = 0;
    int stream_id = 0;

    //loop through all blocks
    for(intT ib = 0; ib < vbmatA.block_rows; ib++ )      //loop vertically through block rows
    {
        rows_in_block = vbmatA.row_part[ib + 1] - vbmatA.row_part[ib]; //the row height of the block
        
        const DataT* d_B_block = d_B + vbmatA.block_col_size*ib;    //access the vertical block of B that is going to be multiplied with blocks of A in block-row ib

        for(intT nzs = 0; nzs < vbmatA.nzcount[ib]; nzs++)        //loop horizontally through nonzero blocks
        {
            
            intT jb = *jab_loc;             //the block row position of a nonzero block 

            DataT* d_C_block = d_C + vbmatA.block_col_size*C_rows*jb;      //access the block on d_C.

            if (stream_col_assign[jb] == -1)
            {
                current_stream++;
                current_stream %= n_streams;
                stream_col_assign[jb] = current_stream;
                stream_id = current_stream;
            }
            else
            {
                stream_id = stream_col_assign[jb];
            }
            cublasSetStream(handle, streams[stream_id]);               //each stream handles a separate block-row
            
            //define the sub-matrices
	        const DataT* d_A_block = d_A + vbmat_idx;           //access the block on d_A.

            int k = rows_in_block, m = B_rows, n = vbmatA.block_col_size;
            int lda = rows_in_block, ldb = B_rows, ldc = C_rows;

            //multiply the blocks, store result in d_C_block
            checkCudaErrors(
                cublasGemmEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m,n,k,                                               //m, n, k <-- block_B: m*k   block_A: k*n   block_C: m*n
                    &alpha,
                    d_B_block,                                          // blockA device pointer,
                    data_type_AB,                                       // blockA datatype
                    ldb,                                      // blockA leading dimension
                    d_A_block,                                          // blockB device pointer
                    data_type_AB,                                       // blockB datatype
                    lda,                                             // B leading dimension
                    &beta,
                    d_C_block, data_type_C,                             // blockC device pointer, blockC type
                    ldc,                                             // C leading dimension
                    compute_type,                                       // compute_type
                    cuda_algo)
            );                                       
            
            //move mab and jab pointers forward
            vbmat_idx += rows_in_block*vbmatA.block_col_size;
            jab_loc++;
	    }

    }

    //record the elapsed time onto dt
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkCudaErrors(cublasGetMatrix(
            C_rows, C_cols, sizeof(DataT_C), d_C, C_rows, C, C_rows));

    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    checkCudaErrors(cublasDestroy(handle));
}

void cublas_blockmat_multiplyBA(const VBR& vbmatA, DataT* B, int B_rows, DataT_C* C, float& dt, int n_streams)
{
//multiplies a VBS matrix (vbmatA) and dense matrix (B); stores B*A into (C)
    //vbmatA:       column-major entries (in-block) storage;
    //              row-major block storage; 
    //B:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    //C:            column-major storage; TODO: allow general storage format (implement through cublas transpose)


    cudaDataType_t data_type_AB;
    cudaDataType_t data_type_C;
    cublasComputeType_t compute_type;

    if (typeid(DataT) == typeid(int8_t))
    {
        data_type_AB = Z_DATATYPE_AB;
        data_type_C =  Z_DATATYPE_C;
        compute_type = Z_CUBLAS_COMPUTE;
    }
    else if (typeid(DataT) == typeid(float))
    {
        data_type_AB = Q_DATATYPE_AB;
        data_type_C =  Q_DATATYPE_C;
        compute_type = Q_CUBLAS_COMPUTE;
    }
    else
    {
        std::cout << "WARNING! Unsopported multiplication type in cublas_blockmat_multiply(). Check matrices.h" << std::endl;
    }


    cublasGemmAlgo_t cuda_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    intT A_rows = vbmatA.rows;
    intT A_cols = vbmatA.cols;
    intT B_cols = A_rows;
    intT C_rows = B_rows;
    intT C_cols = B_cols;

    const DataT_C alpha = 1;
    const DataT_C beta = 1;

    //allocate memory on device
    intT size_A = vbmatA.nztot; //total nonzero entries in vbmat
    intT mem_size_A = sizeof(DataT) * size_A;

    intT size_B = B_rows * B_cols;
    intT mem_size_B = sizeof(DataT) * size_B;

    intT size_C = C_rows * C_cols;
    intT mem_size_C = sizeof(DataT_C) * size_C;

    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    DataT* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy to device the vbmat matrix (nonzero blocks are stored consecutively and in column major format)
    checkCudaErrors(cublasSetVector(
        size_A, sizeof(DataT), vbmatA.mab, 1, d_A, 1));

    //copy B to device (maybe more efficient to copy it block by block?)
    checkCudaErrors(cudaMemcpy(d_B, B, B_rows * B_cols * sizeof(DataT), cudaMemcpyHostToDevice));
    // ----------------------------------------------------------------------

    //creates streams. Each block rows is assigned a different stream.
    
    if (n_streams > vbmatA.block_cols) n_streams = vbmatA.block_cols;
    cudaStream_t streams[n_streams];
    for (intT ib = 0; ib < n_streams; ib++) {
//         cudaStreamCreate(&(streams[ib]));
        cudaStreamCreateWithFlags(&streams[ib],cudaStreamNonBlocking);
    }

    intT mat_idx = 0; //keeps writing position for mat
    intT vbmat_idx = 0; //keeps reading position for vbmat 
    intT ja_count = 0; //keeps total nonzero blocks count;
    intT tot_nonzero_blocks = 0; //index for total nonzero blocks
    intT rows_in_block;
    intT size_block, mem_size_block;
    intT* jab_loc = vbmatA.jab;
    
    //initialize cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);


    intT jb;
    DataT* d_C_block;
    rows_in_block = vbmatA.row_part[1] - vbmatA.row_part[0]; //the row height of the block

    //loop through all blocks
    for(intT ib = 0; ib < vbmatA.block_rows; ib++ )      //loop vertically through block rows
    {
//      rows_in_block = vbmatA.row_part[ib + 1] - vbmatA.row_part[ib]; //the row height of the block
        
        const DataT* d_B_block = d_B + vbmatA.block_col_size*ib;    //access the vertical block of B that is going to be multiplied with blocks of A in block-row ib

        for(intT nzs = 0; nzs < vbmatA.nzcount[ib]; nzs++)        //loop horizontally through nonzero blocks
        {
            
            jb = *jab_loc;             //the block row position of a nonzero block 

            d_C_block = d_C + vbmatA.block_col_size*C_rows*jb;      //access the block on d_C.

            cublasSetStream(handle, streams[jb%n_streams]);               //each stream handles a separate block-row
            
            //define the sub-matrices
	        const DataT* d_A_block = d_A + vbmat_idx;           //access the block on d_A.

//           int k = rows_in_block, m = B_rows, n = vbmatA.block_col_size;
//           int lda = rows_in_block, ldb = B_rows, ldc = C_rows;

            //multiply the blocks, store result in d_C_block
            checkCudaErrors(
                cublasGemmEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    B_rows,vbmatA.block_col_size,rows_in_block,                                               //m, n, k <-- block_B: m*k   block_A: k*n   block_C: m*n
                    &alpha,
                    d_B_block,                                          // blockA device pointer,
                    data_type_AB,                                       // blockA datatype
                    B_rows,                                      // blockA leading dimension
                    d_A_block,                                          // blockB device pointer
                    data_type_AB,                                       // blockB datatype
                    rows_in_block,                                             // B leading dimension
                    &beta,
                    d_C_block, data_type_C,                             // blockC device pointer, blockC type
                    C_rows,                                             // C leading dimension
                    compute_type,                                       // compute_type
                    cuda_algo)
            );                                       
            
            //move mab and jab pointers forward
            vbmat_idx += rows_in_block*vbmatA.block_col_size;
            jab_loc++;
	    }

    }

    //record the elapsed time onto dt
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkCudaErrors(cublasGetMatrix(
            C_rows, C_cols, sizeof(DataT_C), d_C, C_rows, C, C_rows));

    for (int i = 0; i < n_streams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    checkCudaErrors(cublasDestroy(handle));
}

void cublas_blockmat_batched(const VBR& vbmatA, DataT* B, int B_cols, DataT_C* C, float& dt)
{
//multiplies a VBS matrix (vbmatA) and dense matrix (B); stores B*A into (C)
    //vbmatA:       column-major entries (in-block) storage;
    //              row-major block storage; 
    //B:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    //C:            column-major storage; TODO: allow general storage format (implement through cublas transpose)


    cudaDataType_t data_type_AB;
    cudaDataType_t data_type_C;
    cublasComputeType_t compute_type;

    if (typeid(DataT) == typeid(int8_t))
    {
        data_type_AB = Z_DATATYPE_AB;
        data_type_C =  Z_DATATYPE_C;
        compute_type = Z_CUBLAS_COMPUTE;
    }
    else if (typeid(DataT) == typeid(float))
    {
        data_type_AB = Q_DATATYPE_AB;
        data_type_C =  Q_DATATYPE_C;
        compute_type = Q_CUBLAS_COMPUTE;
    }
    else
    {
        std::cout << "WARNING! Unsopported multiplication type in cublas_blockmat_multiply(). Check matrices.h" << std::endl;
    }


    cublasGemmAlgo_t cuda_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    intT A_rows = vbmatA.rows;
    intT A_cols = vbmatA.cols;
    intT B_rows = A_cols;
    intT C_rows = A_rows;
    intT C_cols = B_cols;

    const DataT_C alpha = 1;
    const DataT_C beta = 1;

    //allocate memory on device
    intT size_A = vbmatA.nztot; //total nonzero entries in vbmat
    intT mem_size_A = sizeof(DataT) * size_A;

    intT size_B = B_rows * B_cols;
    intT mem_size_B = sizeof(DataT) * size_B;

    intT size_C = C_rows * C_cols;
    intT mem_size_C = sizeof(DataT_C) * size_C;

    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    DataT *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy to device the vbmat matrix (nonzero blocks are stored consecutively and in column major format)
    checkCudaErrors(cublasSetVector(
        size_A, sizeof(DataT), vbmatA.mab, 1, d_A, 1));

    //copy B to device (maybe more efficient to copy it block by block?)
    checkCudaErrors(cudaMemcpy(d_B, B, B_rows * B_cols * sizeof(DataT), cudaMemcpyHostToDevice));
    // ----------------------------------------------------------------------

    intT tot_nz_blocks = 0;
    intT max_blocks_in_row = 0;
    intT* jab_positions[vbmatA.block_rows];
    DataT* mab_positions[vbmatA.block_rows];
    intT* current_jab = vbmatA.jab;
    DataT* current_mab = d_A;
    intT row_block_size = vbmatA.row_part[1] - vbmatA.row_part[0];
//     std::vector<intT*> jab_positions;
//     std::vector<DataT*> mab_positions;
    for (intT ib = 0; ib < vbmatA.block_rows; ib++)
    {
        assert(vbmatA.row_part[ib+1] - vbmatA.row_part[ib] == row_block_size);
        jab_positions[ib] = (current_jab);
        mab_positions[ib] = (current_mab);
        current_jab += vbmatA.nzcount[ib];
        current_mab += vbmatA.nzcount[ib]*vbmatA.block_col_size*row_block_size;
        max_blocks_in_row = std::max(max_blocks_in_row, vbmatA.nzcount[ib]);
        tot_nz_blocks += vbmatA.nzcount[ib];
    }


    intT block_area = row_block_size*vbmatA.block_col_size;
    DataT** h_d_A = (DataT**) malloc(sizeof(DataT*)*(vbmatA.block_rows));
    DataT** h_d_B = (DataT**) malloc(sizeof(DataT*)*(vbmatA.block_rows));
    DataT** h_d_C = (DataT**) malloc(sizeof(DataT*)*(vbmatA.block_rows));
    DataT **d_A_array, **d_B_array, **d_C_array;
    checkCudaErrors(cudaMalloc((void**)&d_A_array, sizeof(DataT*)*vbmatA.block_rows));
    checkCudaErrors(cudaMalloc((void**)&d_B_array, sizeof(DataT*)*vbmatA.block_rows));
    checkCudaErrors(cudaMalloc((void**)&d_C_array, sizeof(DataT*)*vbmatA.block_rows));
    // Create host pointer array to device matrix storage

   //initialize cuda events
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));

    //loop vertically, find the nth nzs for each block that has one
    for (intT nzs = 0; nzs < max_blocks_in_row; nzs++)
    {

//         h_d_A.clear();
//         h_d_B.clear();
//         h_d_C.clear();

        int batch_cnt = 0;
        for (intT ib = 0; ib < vbmatA.block_rows; ib++)
        {
            if (nzs >= vbmatA.nzcount[ib]) continue;
            intT jb = *(jab_positions[ib] + nzs);
            h_d_A[batch_cnt]=(mab_positions[ib] + nzs*block_area);
            h_d_B[batch_cnt]=(d_B + vbmatA.block_col_size*jb);    //access the vertical block of B that is going to be multiplied with blocks of A in block-row ib
            h_d_C[batch_cnt]=(d_C + vbmatA.row_part[ib]);     //access the block on d_C.
            batch_cnt++;
        }


        //todo make these copies async
        checkCudaErrors(cudaMemcpy(d_A_array, h_d_A, batch_cnt*sizeof(DataT*), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B_array, h_d_B, batch_cnt*sizeof(DataT*), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_C_array, h_d_C, batch_cnt*sizeof(DataT*), cudaMemcpyHostToDevice));

        int k = vbmatA.block_col_size, n = B_cols, m = row_block_size;
        int lda = row_block_size, ldb = B_rows, ldc = C_rows;
//         int lda = block_area, ldb = B_rows, ldc = C_rows;

        checkCudaErrors(cublasSgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       m, n, k,
                       &alpha,
                       (const DataT**) d_A_array, lda,
                       (const DataT**) d_B_array, ldb,
                       &beta,
                       (DataT**) d_C_array, ldc,
                       batch_cnt));
    }

    //record the elapsed time onto dt
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkCudaErrors(cublasGetMatrix(
            C_rows, C_cols, sizeof(DataT_C), d_C, C_rows, C, C_rows));

    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    checkCudaErrors(cublasDestroy(handle));
}


void pico_print_SpMMM(const char* Aname, int An, int Am, int Az, int* Arows, int* Acols, DataT* Avals, const char* Bname, int Bn, int Bm, DataT* B, const char* Cname, long int Cn, long int Cm, DataT_C* C) {

    printf("Dims of %s %s %s:\n", Aname, Bname, Cname);
    printf("       %6s %6s %6s\n", "rows", "cols", "nnz");
    if (Arows != NULL) printf(" %5s %6d %6d %6d\n", Aname, An, Am, Az);
    if (B != NULL)     printf(" %5s %6d %6d %6d\n", Bname, Bn, Bm, 0);
    if (C != NULL)     printf(" %5s %6ld %6ld %6d\n", Cname, Cn, Cm, 0);
    printf("\n");


    if (Arows != NULL) {
        printf("Sparse matrix %s:\n", Aname);
        printf("Rows: ");
        for (int i=0; i<(An+1); i++)
            printf("%3d ", Arows[i]);
        printf("\n");
        printf("Cols: ");
        for (int i=0; i<Az; i++)
            printf("%3d ", Acols[i]);
        printf("\n");
        printf("Vals: ");
        for (int i=0; i<Az; i++)
            printf("%3.2f ", Avals[i]);
        printf("\n\n");
    }

    if (B != NULL) {
        printf("Dense matrix %s:\n", Bname);
        for (int i=0; i<Bn; i++) {
            for (int j=0; j<Bm; j++) {
                printf(" %6.3f ", B[i * Bm + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    if (C != NULL) {
        printf("Dense matrix %s:\n", Cname);
        for (int i=0; i<Cn; i++) {
            for (int j=0; j<Cm; j++) {
                printf(" %6.3f ", C[i * Cm + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

}

void pico_print_SpMMM(const char* Aname, VBR* A, const char* Bname, int Bn, int Bm, DataT* B, const char* Cname, long int Cn, long int Cm, DataT_C* C) {

    printf("Dims of %s %s %s:\n", Aname, Bname, Cname);
    printf("       %6s %6s %6s\n", "rows", "cols", "block_rows x block_cols");
    if (A != NULL)     printf(" %5s %6ld %6ld      %ldx%ld\n", Aname, A->rows, A->cols, A->block_rows, A->block_cols);
    if (B != NULL)     printf(" %5s %6d %6d      %dx%d\n", Bname, Bn, Bm, 0, 0);
    if (C != NULL)     printf(" %5s %6ld %6ld      %dx%d\n", Cname, Cn, Cm, 0, 0);
    printf("\n");


    if (A != NULL) {
        printf("Sparse matrix %s:\n", Aname);
        printf("nzcount: ");
        for (int i=0; i<((A->rows)/(A->block_rows)); i++)
            printf("%3ld ", A->nzcount[i]);
        printf("\n");
        printf("jab: ");
        int k = 0;
        for (int i=0; i<((A->rows)/(A->block_rows)); i++)
            for (int j=0; j<A->nzcount[i]; j++) {
                printf("%3ld ", A->jab[k]);
                k++;
            }
        printf("\n");
        printf("mab: ");
        for (int i=0; i<A->nztot; i++)
            printf("%3.2f ", A->mab[i]);
        printf("\n");
        printf("row_part: ");
        for (int i=0; i<((A->rows)/(A->block_rows) +1); i++)
            printf("%3.2ld ", A->row_part[i]);
        printf("\n\n");
    }

    if (B != NULL) {
        printf("Dense matrix %s:\n", Bname);
        for (int i=0; i<Bn; i++) {
            for (int j=0; j<Bm; j++) {
                printf(" %6.3f ", B[i * Bm + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    if (C != NULL) {
        printf("Dense matrix %s:\n", Cname);
        for (int i=0; i<Cn; i++) {
            for (int j=0; j<Cm; j++) {
                printf(" %6.3f ", C[i * Cm + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

}

void pico_print_DnM(const char* Cname, int Cn, int Cm, DataT_C* C) {

    printf("Dense matrix %s:\n", Cname);
    for (int i=0; i<Cn; i++) {
        for (int j=0; j<Cm; j++) {
            printf(" %6.3f ", C[i * Cm + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void pico_print_SpMMM(const char* Aname, int rows, int cols, int ell_blocksize, int ellValue_cols, int ellColumnsInd_rows, int ellColumnsInd_cols, int num_blocks, intT* ellColumnsInd, DataT_C* ellValues, const char* Bname, int Bn, int Bm, DataT* B, const char* Cname, long int Cn, long int Cm, DataT_C* C) {

    printf("Dims of %s %s %s:\n", Aname, Bname, Cname);
    printf("       %6s %6s\n", "rows", "cols");
    if (ellColumnsInd != NULL)     printf(" %5s %6d %6d\n", Aname, rows, cols);
    if (B != NULL)     printf(" %5s %6d %6d\n", Bname, Bn, Bm);
    if (C != NULL)     printf(" %5s %6ld %6ld\n", Cname, Cn, Cm);
    printf("\n");

    printf("rows = %d, cols = %d, ell_blocksize = %d, ellValue_cols = %d, ellColumnsInd_rows = %d, ellColumnsInd_cols = %d, num_blocks = %d\n", rows, cols, ell_blocksize, ellValue_cols, ellColumnsInd_rows, ellColumnsInd_cols, num_blocks);

    if (ellColumnsInd != NULL) {
        printf("Blocked-ellpac matrix %s:\n", Aname);
        printf("\tellColInd matrix:\n");
        for (int i=0; i<ellColumnsInd_rows; i++) {
            printf("\t\t");
            for (int j=0; j< ellColumnsInd_cols; j++)
                printf("%ld ", ellColumnsInd[i*ellColumnsInd_cols + j]);
            printf("\n");
        }

        printf("\tellValue matrix:\n");
        for (int i=0; i<rows; i++) {
            printf("\t\t");
            for (int j=0; j< ellValue_cols; j++) {
                if ((j%ell_blocksize) == 0)
                    printf("| ");
                if (ellColumnsInd[(i/ell_blocksize)*ellColumnsInd_cols + (j/ell_blocksize)] != -1) {
                    printf("%f ", ellValues[i*ellValue_cols +  j]);
                } else {
                    printf("Pad ");
                }
            }
            printf("\n");
            if ((i%ell_blocksize) == (ell_blocksize-1))
                printf("\t\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        }

        printf("\n\traw ellColInd matrix:\n\t\t");
        for (int i=0; i<ellColumnsInd_cols*ellColumnsInd_rows; i++)
            printf("%ld ", ellColumnsInd[i]);
        printf("\n");

        printf("\traw ellValue matrix:\n\t\t");
        for (int i=0; i<num_blocks*ell_blocksize*ell_blocksize; i++)
            printf("%6.3f ", ellValues[i]);
        printf("\n\n");


    }

    if (B != NULL) {
        printf("Dense matrix %s:\n", Bname);
        for (int i=0; i<Bn; i++) {
            for (int j=0; j<Bm; j++) {
                printf(" %6.3f ", B[i * Bm + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    if (C != NULL) {
        printf("Dense matrix %s:\n", Cname);
        for (int i=0; i<Cn; i++) {
            for (int j=0; j<Cm; j++) {
                printf(" %6.3f ", C[i * Cm + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

}

void ck_out (const char* name, bool test) {
    if (test)
        printf("\t%s test: TRUE\n", name);
    else
        printf("\t%s test: FALSE\n", name);
}

void check_csr(cusparseSpMatDescr_t csr, const char* Aname, int An, int Am, int Az, int* Arows, int* Acols, DataT* Avals) {
    printf("Checking csr %s:\n", Aname);

    int64_t chk_rows;
    int64_t chk_cols;
    int64_t chk_nnz;
    void  *chk_csrRowOffsets, *h_chk_csrRowOffsets;
    void  *chk_csrColInd, *h_chk_csrColInd;
    void  *chk_csrValues, *h_chk_csrValues;
    cusparseIndexType_t  chk_csrColIndType;
    cusparseIndexType_t  chk_csrRowOffsetsType;
    cusparseIndexBase_t  chk_idxBase;
    cudaDataType chk_valueType;


    cusparseCsrGet(
        csr,
        &chk_rows,
        &chk_cols,
        &chk_nnz,
        &chk_csrRowOffsets,
        &chk_csrColInd,
        &chk_csrValues,
        &chk_csrRowOffsetsType,
        &chk_csrColIndType,
        &chk_idxBase,
        &chk_valueType
    );

    bool ck_dim = ((An == chk_rows) && (Am == chk_cols) && (Az == chk_nnz));
    ck_out ("dimension", ck_dim);

    h_chk_csrRowOffsets = malloc(sizeof(int) * An);
    cudaMemcpy(h_chk_csrRowOffsets, chk_csrRowOffsets, sizeof(int) * An, cudaMemcpyDeviceToHost);
    bool ck_row = (memcmp(h_chk_csrRowOffsets, Arows, sizeof(int) * An) == 0) ? 1 : 0;
    ck_out ("rows", ck_row);

    h_chk_csrColInd = malloc(sizeof(int) * Az);
    cudaMemcpy(h_chk_csrColInd, chk_csrColInd, sizeof(int) * Az, cudaMemcpyDeviceToHost);
    bool ck_col = (memcmp(h_chk_csrColInd, Acols, sizeof(int) * Az) == 0) ? 1 : 0;
    ck_out ("cols", ck_col);

    h_chk_csrValues = malloc(sizeof(DataT) * Az);
    cudaMemcpy(h_chk_csrValues, chk_csrValues, sizeof(DataT) * Az, cudaMemcpyDeviceToHost);
    bool ck_val = (memcmp(h_chk_csrValues, Avals, sizeof(DataT) * Az) == 0) ? 1 : 0;
    ck_out ("vals", ck_val);

    printf("\n");

// ---------- They are only readed, not copyed ----------
//     cudaFree(chk_csrRowOffsets);
//     cudaFree(chk_csrColInd);
//     cudaFree(chk_csrValues);
    free(h_chk_csrRowOffsets);
    free(h_chk_csrColInd);
    free(h_chk_csrValues);

    return;
}

void check_bell(cusparseSpMatDescr_t bell, const char* Aname, int rows, int cols, int ell_blocksize, int ellValues_cols, int ellColumnsInd_rows, int ellColumnsInd_cols, int num_blocks, intT* ellColumnsInd, DataT_C* ellValues) {
    printf("Checking blocked ellpack %s:\n", Aname);

    int64_t chk_rows;
    int64_t chk_cols;
    int64_t chk_ell_blocksize;
    int64_t chk_ellValues_cols;
    void* chk_ellColumnsInd, *h_chk_ellColumnsInd;
    void* chk_ellValues, *h_chk_ellValues;
    cusparseIndexType_t chk_ellIdxType;
    cusparseIndexBase_t chk_idxBase;
    cudaDataType chk_valueType;

    cusparseBlockedEllGet(bell,
        &chk_rows,
        &chk_cols,
        &chk_ell_blocksize,
        &chk_ellValues_cols,
        &chk_ellColumnsInd,
        &chk_ellValues,
        &chk_ellIdxType,
        &chk_idxBase,
        &chk_valueType);

    bool ck_dim = ((rows == chk_rows) && (cols == chk_cols) && (ell_blocksize == chk_ell_blocksize) && (ellValues_cols == chk_ellValues_cols)), ck_col, ck_val;
    ck_out ("dimension", ck_dim);

    if (ck_dim) {
        h_chk_ellColumnsInd = malloc(sizeof(intT) * (ellColumnsInd_cols*ellColumnsInd_rows));
        cudaMemcpy(h_chk_ellColumnsInd, chk_ellColumnsInd, sizeof(intT) * (ellColumnsInd_cols*ellColumnsInd_rows), cudaMemcpyDeviceToHost);
        ck_col = (memcmp(h_chk_ellColumnsInd, ellColumnsInd, sizeof(intT) * (ellColumnsInd_cols*ellColumnsInd_rows)) == 0) ? 1 : 0;
        ck_out ("ellColumnsInd_cols", ck_col);
    } else {
        printf("(rows == chk_rows): (%d, %ld) && (cols == chk_cols): (%d, %ld) && (ell_blocksize == chk_ell_blocksize): (%d, %ld) && (ellValues_cols == chk_ellValues_cols): (%d, %ld)\n", rows, chk_rows, cols, chk_cols, ell_blocksize, chk_ell_blocksize, ellValues_cols, chk_ellValues_cols);
    }

    if (ck_dim) {
        h_chk_ellValues = malloc(sizeof(DataT_C) * (num_blocks*ell_blocksize*ell_blocksize));
        cudaMemcpy(h_chk_ellValues, chk_ellValues, sizeof(DataT_C) * (num_blocks*ell_blocksize*ell_blocksize), cudaMemcpyDeviceToHost);
        ck_val = (memcmp(h_chk_ellValues, ellValues, sizeof(DataT_C) * (num_blocks*ell_blocksize*ell_blocksize)) == 0) ? 1 : 0;
        ck_out ("ell_val", ck_val);
    }

    if (ck_col == false || ck_val == false) {
        printf("h_chk_matrix:\n");
        pico_print_SpMMM("BEL_A", chk_rows, chk_cols, chk_ell_blocksize, chk_ellValues_cols, ellColumnsInd_rows, ellColumnsInd_cols, num_blocks, (intT*)h_chk_ellColumnsInd, (DataT_C*)h_chk_ellValues, "NULL", 0, 0, NULL, "NULL", 0, 0, NULL);
    }

    printf("\n");

// ---------- They are only readed, not copyed ----------
//     cudaFree(chk_ellColumnsInd);
//     cudaFree(chk_ellValues);
    free(h_chk_ellColumnsInd);
    free(h_chk_ellValues);

    return;
}

void check_dnmat(cusparseDnMatDescr_t dnmat, const char* Bname, int Bn, int Bm, DataT* B) {
    printf("Checking dense matrix %s:\n", Bname);


    int64_t chk_rows;
    int64_t chk_cols;
    int64_t chk_ld;
    void  *chk_vals, *h_chk_vals;
    cudaDataType chk_valueType;
    cusparseOrder_t chk_order;

    cusparseDnMatGet(dnmat,
        &chk_rows,
        &chk_cols,
        &chk_ld,
        &chk_vals,
        &chk_valueType,
        &chk_order
    );



    bool ck_dim = ((Bn == chk_rows) && (Bm == chk_cols) && (Bm == chk_ld));
    ck_out ("dimension", ck_dim);

    h_chk_vals = malloc(sizeof(DataT) * Bn * Bm);
    cudaMemcpy(h_chk_vals, chk_vals, sizeof(DataT) * Bn * Bm, cudaMemcpyDeviceToHost);
    bool ck_val = (memcmp(h_chk_vals, B, sizeof(DataT) * Bn * Bm) == 0) ? 1 : 0;
    ck_out ("vals", ck_val);

    printf("\n");


// ---------- They are only readed, not copyed ----------
//     cudaFree(chk_vals);
    free(h_chk_vals);

    return;
}

int cusparse_gemm_custom(int rows, int cols, int nnz, int* csrRowPtr, int* csrColInd, DataT* csrVal, DataT* B, int B_cols, int B_lead_dim, DataT_C* C, int C_lead_dim, const DataT_C alpha, const DataT_C beta, float& dt)
{

    cudaDataType_t data_type_AB;
    cudaDataType_t data_type_C;
    cudaDataType_t compute_type;

    if (typeid(DataT) == typeid(int8_t))
    {
        data_type_AB = Z_DATATYPE_AB;
        data_type_C =  Z_DATATYPE_C;
    }
    else if (typeid(DataT) == typeid(float))
    {
        data_type_AB = Q_DATATYPE_AB;
        data_type_C =  Q_DATATYPE_C;
        compute_type = CUDA_R_32F;    // NOTE: CUDA_R_16F not supported
    }
    else
    {
        std::cout << "WARNING! Unsopported multiplication type. Check comp_mats.h" << std::endl;
    }


    //allocate memory on device
    unsigned int mem_size_csrVal = sizeof(DataT) * nnz;
    unsigned int mem_size_csrColInd = sizeof(int) * nnz;
    unsigned int mem_size_csrRowPtr = sizeof(int) * (rows + 1);

    unsigned int B_rows = cols;
    unsigned int size_B = B_rows * B_cols;
    unsigned int mem_size_B = sizeof(DataT) * size_B;

    long int C_rows = rows;
    long int C_cols = B_cols;
    unsigned int size_C = C_rows * C_cols;
    unsigned int mem_size_C = sizeof(DataT_C) * size_C;

    // allocate device memory
    int *d_RowPtr, *d_ColInd;
    DataT *d_Val;

#ifdef PICO_DEBUG
    printf("alpha = %f, beta = %f\n", alpha, beta);
    pico_print_SpMMM("A", rows, cols, nnz, csrRowPtr, csrColInd, csrVal, "B", B_rows, B_cols, B, "C", C_rows, C_cols, C);
#endif

    checkCudaErrors(cudaMalloc((void**)&d_RowPtr, mem_size_csrRowPtr));
    checkCudaErrors(cudaMalloc((void**)&d_ColInd, mem_size_csrColInd));
    checkCudaErrors(cudaMalloc((void**)&d_Val, mem_size_csrVal));

    DataT* d_B;
    DataT_C* d_C;
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy arrays and matrices to device
    checkCudaErrors( cudaMemcpy(d_Val, csrVal, mem_size_csrVal, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_ColInd, csrColInd, mem_size_csrColInd, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_RowPtr, csrRowPtr, mem_size_csrRowPtr, cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice) );

    if (beta != 0)
    {
        checkCudaErrors( cudaMemcpy(d_C, C, mem_size_C, cudaMemcpyHostToDevice) );
    }

    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    checkCudaErrors(cusparseCreate(&handle));

    checkCudaErrors(
        cusparseCreateCsr(
            &matA,
            rows,
            cols,
            nnz,
            d_RowPtr,
            d_ColInd,
            d_Val,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            data_type_AB
            )
        );

#ifdef PICO_DEBUG
    check_csr(matA, "A", rows, cols, nnz, csrRowPtr, csrColInd, csrVal);
#endif

    checkCudaErrors(
        cusparseCreateDnMat(&matB, cols, B_cols, B_lead_dim, d_B,
            data_type_AB, CUSPARSE_ORDER_ROW));

#ifdef PICO_DEBUG
    check_dnmat(matB, "matB", B_rows, B_cols, B);
#endif

    checkCudaErrors(
        cusparseCreateDnMat(&matC, rows, B_cols, C_lead_dim, d_C,
            data_type_C, CUSPARSE_ORDER_ROW));

#ifdef PICO_DEBUG
    check_dnmat(matC, "matC", C_rows, C_cols, C);
#endif



    size_t bufferSize = 0;
    void *dBuffer = NULL;

    checkCudaErrors(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        (void*)&alpha,
        matA,
        matB,
        (void*)&beta,
        matC,
        compute_type,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize
    ));
    checkCudaErrors( cudaMalloc(&dBuffer, bufferSize) );


    //initialize cuda events
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));


    checkCudaErrors(cusparseSpMM(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, compute_type,
        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));       // We have a BUG here

    //record the elapsed time onto dt
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // destroy matrix/vector descriptors
    checkCudaErrors(cusparseDestroySpMat(matA));
    checkCudaErrors(cusparseDestroyDnMat(matB));
    checkCudaErrors(cusparseDestroyDnMat(matC));

    // copy result from device to host
    // -------------------------------------------------
//     checkCudaErrors(cublasGetMatrix(C_rows, C_cols, sizeof(DataT_C), d_C, C_rows, C, C_rows));
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    checkCudaErrors(cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    // -------------------------------------------------

#ifdef PICO_DEBUG
    printf("Printing results of %s:\n", __func__);
    pico_print_SpMMM("A", rows, cols, nnz, csrRowPtr, csrColInd, csrVal, "B", B_rows, B_cols, B, "C", C_rows, C_cols, C);
#endif

    // clean up memor;y
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_Val));
    checkCudaErrors(cudaFree(d_RowPtr));
    checkCudaErrors(cudaFree(d_ColInd));

    // Destroy the handle
    checkCudaErrors(cusparseDestroy(handle));

    return 0;
}

int prepare_cusparse_CSR(CSR& cmat, int** csrRowPtr, int** csrColInd, DataT** csrVal)
{
//     if (cmat.fmt != 0)
//     {
//         std::cout << "ERROR: cusparse_gemm_custom only supports CSR (row-major) " << std::endl;
//         return 1;
//     }

#ifdef PICO_DEBUG
    printf("File %s, line %d: print cmat\n", __FILE__, __LINE__);
    cmat.print(2);
#endif

    *csrRowPtr = (int*) malloc( (cmat.rows+1) * sizeof(int));

    //fill csrRowPtr (element i holds number of nonzero entries up to row i)
    (*csrRowPtr)[0] = 0;

    int nnz = 0;
    for (int i = 0; i < cmat.rows; i++)
    {
        nnz += cmat.nzcount[i];
        (*csrRowPtr)[i + 1] = nnz;
    }
    //-------------------------------------------------------------

    *csrColInd = (int*) malloc( nnz * sizeof(int));
    *csrVal = (DataT*) malloc( nnz * sizeof(DataT));

    //fill csrVal (the nonzero values) and csrColInd (their column indices)
    nnz = 0;
    for (int i = 0; i < cmat.rows; i++)
    {
        std::copy(cmat.ja[i], cmat.ja[i] + cmat.nzcount[i], (*csrColInd) + nnz);
        if (cmat.pattern_only == 0)
            std::copy(cmat.ma[i], cmat.ma[i] + cmat.nzcount[i], (*csrVal) + nnz);
        else
            for (int j=0; j<cmat.nzcount[i]; j++)
                (*csrVal)[nnz +j] = 1;
        nnz += cmat.nzcount[i];

    }

    return 0;
}

void cusparse_blockmat_multiplyAB(CSR& A, DataT* B, int B_cols, DataT_C* C, int C_cols, float& dt) {

    DataT *csrVal;
    int *csrRowPtr, *csrColInd;

    prepare_cusparse_CSR( A, &csrRowPtr, &csrColInd, &csrVal);

    cusparse_gemm_custom(A.rows, A.cols, (int) A.nztot(), csrRowPtr, csrColInd, csrVal, B, B_cols, B_cols, C, C_cols, 1, 1, dt);

    free(csrVal);
    free(csrColInd);
    free(csrRowPtr);

    return;
}



int cusparse_gemm_custom_ellpack(int rows, int cols, int A_ell_blocksize, int A_ellValues_cols, int A_ellColInd_cols, int A_ellColInd_rows, int A_num_blocks, intT* A_ellColInd, DataT_C* A_ellValues, DataT* B, int B_cols, int B_lead_dim, DataT_C* C, int C_lead_dim, const DataT_C alpha, const DataT_C beta, float& dt)
{

    cudaDataType_t data_type_AB;
    cudaDataType_t data_type_C;
    cudaDataType_t compute_type;

    if (typeid(DataT) == typeid(int8_t))
    {
        data_type_AB = Z_DATATYPE_AB;
        data_type_C =  Z_DATATYPE_C;
    }
    else if (typeid(DataT) == typeid(float))
    {
        data_type_AB = Q_DATATYPE_AB;
        data_type_C =  Q_DATATYPE_C;
        compute_type = Q_CUSPARSE_COMPUTE;
    }
    else
    {
        std::cout << "WARNING! Unsopported multiplication type. Check comp_mats.h" << std::endl;
    }


    //allocate memory on device


    unsigned int B_rows = cols;
    unsigned int size_B = B_rows * B_cols;
    unsigned int mem_size_B = sizeof(DataT) * size_B;

// TO ADAPT

    long int C_rows = rows;
    long int C_cols = B_cols;
    unsigned int size_C = C_rows * C_cols;
    unsigned int mem_size_C = sizeof(DataT_C) * size_C;

    // allocate device memory
    intT    *dA_ellColInd;
    DataT_C *dA_ellValues;


#ifdef PICO_DEBUG
    printf("alpha = %f, beta = %f\n", alpha, beta);
//     pico_print_SpMMM("A", rows, cols, nnz, csrRowPtr, csrColInd, csrVal, "B", B_rows, B_cols, B, "C", C_rows, C_cols, C);
#endif

    checkCudaErrors( cudaMalloc((void**) &dA_ellColInd, A_ellColInd_cols * A_ellColInd_rows * sizeof(intT)) );
    checkCudaErrors( cudaMalloc((void**) &dA_ellValues, A_ell_blocksize * A_ell_blocksize * A_num_blocks * sizeof(DataT_C)) );

    DataT* d_B;
    DataT_C* d_C;
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy arrays and matrices to device
    checkCudaErrors( cudaMemcpy(dA_ellColInd, A_ellColInd, A_ellColInd_rows * A_ellColInd_cols * sizeof(intT), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dA_ellValues, A_ellValues, A_num_blocks * A_ell_blocksize * A_ell_blocksize * sizeof(DataT_C), cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice) );

    if (beta != 0)
    {
        checkCudaErrors( cudaMemcpy(d_C, C, mem_size_C, cudaMemcpyHostToDevice) );
    }

    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    checkCudaErrors(cusparseCreate(&handle));

    checkCudaErrors( cusparseCreateBlockedEll(
                                      &matA,
                                      rows, cols, A_ell_blocksize,
                                      A_ellValues_cols, dA_ellColInd, dA_ellValues,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, data_type_AB) );

#ifdef PICO_DEBUG
    check_bell(matA, "A", rows, cols, A_ell_blocksize, A_ellValues_cols, A_ellColInd_rows, A_ellColInd_cols, A_num_blocks, A_ellColInd, A_ellValues);
#endif

    checkCudaErrors(
        cusparseCreateDnMat(&matB, cols, B_cols, B_lead_dim, d_B,
            data_type_AB, CUSPARSE_ORDER_ROW));

#ifdef PICO_DEBUG
    check_dnmat(matB, "matB", B_rows, B_cols, B);
#endif

    checkCudaErrors(
        cusparseCreateDnMat(&matC, rows, B_cols, C_lead_dim, d_C,
            data_type_C, CUSPARSE_ORDER_ROW));

#ifdef PICO_DEBUG
    check_dnmat(matC, "matC", C_rows, C_cols, C);
#endif



    size_t bufferSize = 0;
    void *dBuffer = NULL;

    checkCudaErrors( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, compute_type,
                                 CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &bufferSize) );
    checkCudaErrors( cudaMalloc(&dBuffer, bufferSize) );


    //initialize cuda events
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    checkCudaErrors( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, compute_type,
                                CUSPARSE_SPMM_BLOCKED_ELL_ALG1, dBuffer) );

    //record the elapsed time onto dt
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // destroy matrix/vector descriptors
    checkCudaErrors(cusparseDestroySpMat(matA));
    checkCudaErrors(cusparseDestroyDnMat(matB));
    checkCudaErrors(cusparseDestroyDnMat(matC));

    // copy result from device to host
    // -------------------------------------------------
//     checkCudaErrors(cublasGetMatrix(C_rows, C_cols, sizeof(DataT_C), d_C, C_rows, C, C_rows));
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    checkCudaErrors(cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    // -------------------------------------------------

    // clean up memor;y
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(dA_ellValues));
    checkCudaErrors(cudaFree(dA_ellColInd));

    // Destroy the handle
    checkCudaErrors(cusparseDestroy(handle));

    return 0;
}


int prepare_cusparse_BLOCKEDELLPACK(VBR *A, int *ell_blocksize, int *ellValue_cols, int *ellColInd_rows, int *ellColInd_cols, int *num_blocks, intT** ellColInd, DataT_C** ellValues)
{
//     if (cmat.fmt != 0)
//     {
//         std::cout << "ERROR: cusparse_gemm_custom only supports CSR (row-major) " << std::endl;
//         return 1;
//     }
    *ell_blocksize = A->block_col_size;
    if ((A->rows)%(*ell_blocksize) != 0 || (A->cols)%(*ell_blocksize)) {
        if ((A->rows)%(*ell_blocksize) != 0)
            printf("The number of rows is not multiple of ell_blocksize\n");
        else
            printf("The number of cols is not multiple of ell_blocksize\n");
        exit(__LINE__);
    }

    *ellColInd_rows = ((A->rows)/(*ell_blocksize));
    *ellColInd_cols = 0;
    for (int i=0; i<(*ellColInd_rows); i++) {
        if (A->nzcount[i] > (*ellColInd_cols))
            (*ellColInd_cols) = A->nzcount[i];
    }
    *ellValue_cols = (*ellColInd_cols) * (*ell_blocksize);

    *num_blocks = (*ellColInd_rows)*(*ellColInd_cols);
    *ellColInd = (intT*)malloc(sizeof(intT)*(*ellColInd_cols)*(*ellColInd_rows));
    *ellValues = (DataT_C*)malloc(sizeof(DataT)*((*num_blocks))*(*ell_blocksize)*(*ell_blocksize));

    int k_col = 0, k, i, j, vbr_blk_row_shift, bel_blk_row_shift;

    // to complete ellColInd
    for (i=0; i<(*ellColInd_rows); i++)
        for (j=0; j<(*ellColInd_cols); j++)
            if (j < A->nzcount[i]) {
                (*ellColInd)[i*(*ellColInd_cols)+j] = A->jab[k_col];
                k_col++;
            } else {
                (*ellColInd)[i*(*ellColInd_cols)+j] = -1; // '-1' means thet the algorithm autmaticly pad it with a compleate zero block
            }

    // to complete ellValues
    vbr_blk_row_shift = 0;
    bel_blk_row_shift = 0;
    for(k=0; k<(*ellColInd_rows); k++) {
        for (i=0; i<(*ell_blocksize); i++)
            for (j=0; j<(*ellValue_cols); j++)
                (*ellValues)[bel_blk_row_shift + i*(*ellValue_cols) + j] =
                    ((*ellColInd)[k*(*ellColInd_cols) + (j/(*ell_blocksize))] != -1) ? A->mab[vbr_blk_row_shift + j*(*ell_blocksize) + i] : 0.0;

        vbr_blk_row_shift += (A->nzcount[k])*(*ell_blocksize)*(*ell_blocksize);
        bel_blk_row_shift += (*ellColInd_cols)*(*ell_blocksize)*(*ell_blocksize);
    }

    return 0;
}

void bellpack_blockmat_multiplyAB(VBR* A, DataT* B, int B_cols, DataT_C* C, int C_cols, float& dt, int verbose) {

        // ellValue_cols, int *ell_blocksize, int *ellColInd_rows, int *ellColInd_cols, int *num_blocks, intT** ellColInd, DataT_C** ellValues
        int ell_blocksize, ellColInd_rows, ellColInd_cols, ellValue_cols, num_blocks;
        intT* ellColInd;
        DataT_C* ellValues;
        prepare_cusparse_BLOCKEDELLPACK(A, &ell_blocksize, &ellValue_cols, &ellColInd_rows, &ellColInd_cols, &num_blocks, &ellColInd, &ellValues);

        if (verbose > 1) {
            int pad_num = 0;
            for (int i=0; i<ellColInd_rows*ellColInd_cols; i++)
                if (ellColInd[i] == -1)
                    pad_num++;
            printf("ell_blocksize = %d, ellColInd has dimensions %d x %d with %d padding blocks, ellValues has dimensions %ld x %d\n", ell_blocksize, ellColInd_rows, ellColInd_cols, pad_num, A->rows, ellValue_cols);
        }

        cusparse_gemm_custom_ellpack(A->rows, A->cols, ell_blocksize, ellValue_cols, ellColInd_cols, ellColInd_rows, num_blocks, ellColInd, ellValues, B, B_cols, B_cols, C, C_cols, 1, 1, dt);

        free(ellColInd);
        free(ellValues);

    return;
}

DataT* csr2dn (CSR& A) {
    if (A.rows == 0) {
        fprintf(stderr, "Error in %s %d.\n", __func__, __LINE__);
        fprintf(stderr, "rows %ld cols %ld.\n", A.rows, A.cols);
        return(NULL);
    }

    int n = A.rows, m = A.cols;
    DataT *csrVal;
    int *csrRowPtr, *csrColInd;

    prepare_cusparse_CSR( A, &csrRowPtr, &csrColInd, &csrVal);
    cudaDeviceSynchronize();

    DataT* dn_A = (DataT*)malloc(sizeof(DataT)*(n)*(m));

    int csr_j;
    for (int i=0; i<n; i++) {
        csr_j = csrRowPtr[i];
        for (int j=0; j<m; j++) {
            if (csr_j < csrRowPtr[i + 1] && j==csrColInd[csr_j]) {
                dn_A[i*m + j] = csrVal[csr_j];
                csr_j ++;
            } else {
                dn_A[i*m + j] = (DataT)0;
            }

        }
    }

    return(dn_A);
}

void cublas_dense_multiplyAB(int rows, int cols, DataT* A, DataT* B, int B_cols, DataT_C* C, float& dt) {
//multiplies a VBS matrix (vbmatA) and dense matrix (B); stores A*B into (C)
    //vbmatA:       column-major entries (in-block) storage;
    //              row-major block storage;
    //B:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    //C:            column-major storage; TODO: allow general storage format (implement through cublas transpose)


    cudaDataType_t data_type_AB;
    cudaDataType_t data_type_C;
    cublasComputeType_t compute_type;

    if (typeid(DataT) == typeid(int8_t))
    {
        data_type_AB = Z_DATATYPE_AB;
        data_type_C =  Z_DATATYPE_C;
        compute_type = Z_CUBLAS_COMPUTE;
    }
    else if (typeid(DataT) == typeid(float))
    {
        data_type_AB = Q_DATATYPE_AB;
        data_type_C =  Q_DATATYPE_C;
        compute_type = Q_CUBLAS_COMPUTE;
    }
    else
    {
        std::cout << "WARNING! Unsopported multiplication type in cublas_blockmat_multiply(). Check matrices.h" << std::endl;
    }


    cublasGemmAlgo_t cuda_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    intT A_rows = rows;
    intT A_cols = cols;

    intT B_rows = A_cols;
    intT C_rows = A_rows;
    int C_cols = B_cols;

    const DataT_C alpha = 1;
    const DataT_C beta = 1;

    //allocate memory on device
    intT size_A = A_rows * A_cols; //total nonzero entries in vbmat
    intT mem_size_A = sizeof(DataT) * size_A;

    intT size_B = B_rows * B_cols;
    intT mem_size_B = sizeof(DataT) * size_B;

    intT size_C = C_rows * C_cols;
    intT mem_size_C = sizeof(DataT_C) * size_C;

    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    DataT* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy A to device (maybe more efficient to copy it block by block?)
    checkCudaErrors(cudaMemcpy(d_A, A, A_rows * A_cols * sizeof(DataT), cudaMemcpyHostToDevice));

    //copy B to device (maybe more efficient to copy it block by block?)
    checkCudaErrors(cudaMemcpy(d_B, B, B_rows * B_cols * sizeof(DataT), cudaMemcpyHostToDevice));
    // ----------------------------------------------------------------------

    //initialize cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //multiply A with B, store result in d_C_block

    checkCudaErrors(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            A_rows,B_cols,A_cols,                                               //m, n, k <-- block_A: m*k   block_B: k*n   block_C: m*n
            &alpha,
            d_A,                                          // blockA device pointer,
            data_type_AB,                                       // blockA datatype
            A_cols,                                      // blockA leading dimension
            d_B,                                          // blockB device pointer
            data_type_AB,                                       // blockB datatype
            B_rows,                                             // B leading dimension
            &beta,
            d_C, data_type_C,                             // blockC device pointer, blockC type
            C_rows,                                             // C leading dimension
            compute_type,                                       // compute_type
            cuda_algo)
    );

    //record the elapsed time onto dt
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    //let each stream copy the relevant C block from device
    // --------------------------------------------------------------
    checkCudaErrors(cublasGetMatrix(
        C_rows, C_cols, sizeof(DataT_C), d_C, C_rows, C, C_rows));
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//     checkCudaErrors(cudaMemcpy(C, d_C, C_rows * C_cols * sizeof(DataT), cudaMemcpyDeviceToHost));
    // --------------------------------------------------------------

    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    checkCudaErrors(cublasDestroy(handle));
}
