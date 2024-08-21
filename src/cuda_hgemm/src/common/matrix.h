#pragma once

#include <random>
#include <tuple>
#include <vector>
#include <iostream>
#include <string>
#include <fstream> 

#include "common.h"
#include "mmio_highlevel.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16


class Matrix {
public:
    Matrix(size_t row, size_t col, const std::string &name = "Matrix", float min = -1.0, float max = 1.0)
        : m_row(row), m_col(col), m_name(name), m_min(min), m_max(max) {
        HGEMM_CHECK_GT(m_row, 0);
        HGEMM_CHECK_GT(m_col, 0);

        m_elem_num = m_row * m_col;
        HGEMM_CHECK_GT(m_elem_num, 0);

        m_host_ptr = new half[m_elem_num];
        HGEMM_CHECK(m_host_ptr);
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(half)));
        HGEMM_CHECK(m_dev_ptr);

        std::random_device rd;
        std::default_random_engine engine{rd()};
        std::uniform_real_distribution<float> uniform(m_min, m_max);
        for (size_t i = 0; i < m_elem_num; ++i) {
            //m_host_ptr[i] = __float2half(uniform(engine));
            m_host_ptr[i] = __float2half(1);
        }

        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice));

        HLOG("%s: %zu * %zu, cpu: %p, gpu: %p", m_name.c_str(), m_row, m_col, m_host_ptr, m_dev_ptr);
    }

    ~Matrix() {
        if (m_host_ptr) {
            delete[] m_host_ptr;
            m_host_ptr = nullptr;
        }

        if (m_dev_ptr) {
            HGEMM_CHECK_CUDART_ERROR(cudaFree((void *)m_dev_ptr));
            m_dev_ptr = nullptr;
        }
    }

    size_t getRow() const {
        return m_row;
    }

    size_t getCol() const {
        return m_col;
    }

    size_t getElemNum() const {
        return m_elem_num;
    }

    half *getHostPtr() const {
        return m_host_ptr;
    }

    half *getDevPtr() const {
        return m_dev_ptr;
    }

    void tearUp(Matrix *base) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());

        HGEMM_CHECK_CUDART_ERROR(
            cudaMemcpy(m_dev_ptr, base->getDevPtr(), m_elem_num * sizeof(half), cudaMemcpyDeviceToDevice));
    }

    void moveToHost() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_host_ptr, m_dev_ptr, m_elem_num * sizeof(half), cudaMemcpyDeviceToHost));
    }

    void moveToDevice() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice));
    }

    void memSetHost() {
        memset(m_host_ptr, 0, m_elem_num * sizeof(half));
    }

    void memSetDevice() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(half)));
    }

    void checkValue(Matrix *base) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());

        m_max_diff = 0.0;
        m_avg_diff = 0.0;
        double diff = 0.0;
        for (size_t i = 0; i < m_elem_num; ++i) {
            diff = static_cast<double>(std::abs(__half2float(m_host_ptr[i]) - __half2float(base->getHostPtr()[i])));
            m_max_diff = std::max(m_max_diff, diff);
            m_avg_diff += diff;
            //HLOG("%f", diff);
        }
        //HLOG("%d", m_elem_num);
        m_avg_diff /= static_cast<double>(m_elem_num);

        HLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    }

private:
    const size_t m_row = 0;
    const size_t m_col = 0;
    const std::string m_name = "Matrix";
    // the threshold of the random matrix will affect the difference of the hgemm results
    const float m_min = -1.0;
    const float m_max = 1.0;

    size_t m_elem_num = 0;
    half *m_host_ptr = nullptr;
    half *m_dev_ptr = nullptr;

    double m_max_diff = 0.0;
    double m_avg_diff = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(Matrix);
};



class SparseMatrix {
public:
    SparseMatrix(const std::string &name = "Matrix", char* file = "./data/cop20k_A.mtx")
        : m_name(name), filename(file) {

        readCsr();

        HLOG("Read %s", filename);
        //outputCsr();
        HGEMM_CHECK_GT(m_row, 0);
        HGEMM_CHECK_GT(m_col, 0);
        HGEMM_CHECK_GT(nnz, 0);

        HLOG("%zu x %zu, nnz = %zu, A[0] = %f", m_row, m_col, nnz, __half2float(csrVal_host[0]));

        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&csrVal_dev, nnz * sizeof(half)));
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&csrColIdx_dev, nnz * sizeof(int)));
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&csrRowPtr_dev, (m_row + 1) * sizeof(int)));

        HGEMM_CHECK(csrVal_dev);
        HGEMM_CHECK(csrColIdx_dev);
        HGEMM_CHECK(csrRowPtr_dev);

        //HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice));

        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(csrVal_dev, csrVal_host, nnz * sizeof(half), cudaMemcpyHostToDevice));
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(csrColIdx_dev, csrColIdx_host, nnz * sizeof(int), cudaMemcpyHostToDevice));
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(csrRowPtr_dev, csrRowPtr_host, (m_row + 1) * sizeof(int), cudaMemcpyHostToDevice));


        csrToBcsr();
        //csrToBcsrKnapsacking();
        HLOG("Finished creating BCSR from CSR");
        HLOG("%zu total blocks, %zu nonzero blocks, %zu dense blocks, %zu sparse blocks", numberOfBlocks, nonzeroBlocks, denseBlocks, sparseBlocks);
        // HLOG("%d block is nonzero", blockInfo[0]);
        /* for (int i = 0; i < MMA_M; i++) {
            for (int j = 0; j < MMA_K; j++) {
                printf("%4.2f ", __half2float(bcsrVal_host[i * MMA_K + j]));
            }
            printf("\n");
        } */

        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&bcsrVal_dev, nonzeroBlocks * MMA_M * MMA_K * sizeof(half)));
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&bcsrColIdx_dev, nonzeroBlocks * sizeof(int)));
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&bcsrRowPtr_dev, (m_row / MMA_M + 1) * sizeof(int)));
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&blockInfo_dev, numberOfBlocks * sizeof(int)));
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&relativeBlockIndexMapping_dev, numberOfBlocks * sizeof(int)));

        HGEMM_CHECK(bcsrVal_dev);
        HGEMM_CHECK(bcsrColIdx_dev);
        HGEMM_CHECK(bcsrRowPtr_dev);

        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(bcsrVal_dev, bcsrVal_host, nonzeroBlocks * MMA_M * MMA_K * sizeof(half), cudaMemcpyHostToDevice));
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(bcsrColIdx_dev, bcsrColIdx_host, nonzeroBlocks * sizeof(int), cudaMemcpyHostToDevice));
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(bcsrRowPtr_dev, bcsrRowPtr_host, (m_row / MMA_M + 1) * sizeof(int), cudaMemcpyHostToDevice));
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(blockInfo_dev, blockInfo_host, numberOfBlocks * sizeof(int), cudaMemcpyHostToDevice));
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(relativeBlockIndexMapping_dev, relativeBlockIndexMapping_host, numberOfBlocks * sizeof(int), cudaMemcpyHostToDevice));

        
    }

    ~SparseMatrix() {
        if (m_host_ptr) {
            delete[] m_host_ptr;
            m_host_ptr = nullptr;
        }

        if (m_dev_ptr) {
            HGEMM_CHECK_CUDART_ERROR(cudaFree((void *)m_dev_ptr));
            m_dev_ptr = nullptr;
        }
    }

    size_t getRow() {
        return m_row;
    }

    size_t getCol() {
        return m_col;
    }

    size_t getElemNum() {
        return m_elem_num;
    }

    size_t getNnz() {
        return nnz;
    }

    half *getHostPtr() {
        return m_host_ptr;
    }

    half *getDevPtr() {
        return m_dev_ptr;
    }

    half *getBcsrValues() {
        return bcsrVal_dev;
    }

    int *getBcsrRowPtr() {
        return bcsrRowPtr_dev;
    }

    int *getBcsrColIdx() {
        return bcsrColIdx_dev;
    }

    int *getRelativeBlockIndexMapping_dev() {
        return relativeBlockIndexMapping_dev;
    }

    size_t getNonzeroblocks() {
        return nonzeroBlocks;
    }

    int *getBlockInfo_dev() {
        return blockInfo_dev;
    }

    int *getBlockInfo_host() {
        return blockInfo_host;
    }


    void tearUp(Matrix *base) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());

        HGEMM_CHECK_CUDART_ERROR(
            cudaMemcpy(m_dev_ptr, base->getDevPtr(), m_elem_num * sizeof(half), cudaMemcpyDeviceToDevice));
    }

    void moveToHost() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_host_ptr, m_dev_ptr, m_elem_num * sizeof(half), cudaMemcpyDeviceToHost));
    }

    void moveToDevice() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice));
    }

    void memSetHost() {
        memset(m_host_ptr, 0, m_elem_num * sizeof(half));
    }

    void memSetDevice() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(half)));
    }

    void checkValue(Matrix *base) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());
        if (m_elem_num == 0) {
            m_elem_num = m_row * m_col;
        }
        m_max_diff = 0.0;
        m_avg_diff = 0.0;
        double diff = 0.0;
        for (size_t i = 0; i < m_elem_num; ++i) {
            diff = static_cast<double>(std::abs(__half2float(m_host_ptr[i]) - __half2float(base->getHostPtr()[i])));
            m_max_diff = std::max(m_max_diff, diff);
            m_avg_diff += diff;
        }

        m_avg_diff /= static_cast<double>(m_elem_num);

        HLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    }

    void readCsr() {
        int isSymmetric;
        mmio_allinone(&m_row, &m_col, &nnz, &isSymmetric, &csrRowPtr_host, &csrColIdx_host, &csrVal_host, filename);

    }

    void outputCsr() {
        // Open file_transformed.mtx for writing
        std::string file_name_str(filename);
        std::ofstream outfile("./data/magicube_cop20k_A.mtx");

        // Check if file is opened successfully
        if (!outfile) {
            std::cerr << "Failed to open file_transformed.mtx for writing." << std::endl;
            return; // Exit with error
        }

        // Write the values of a, b, c separated by commas in the first line
        outfile << m_row << ", " << m_col << ", " << nnz << std::endl;

        // Write the elements of array t separated by spaces in the second line
        for (int i = 0; i < m_row + 1; ++i) {
            outfile << csrRowPtr_host[i] << " ";
        }
        outfile << std::endl;

        for (int i = 0; i < nnz; ++i) {
            outfile << csrColIdx_host[i] << " ";
        }
        outfile << std::endl;
        // Close the file
        outfile.close();
        HLOG("Outputed to Magicube format.");
    }

    void makeDenseArray() {
        m_elem_num = m_row * m_col;
        m_host_ptr = new half[m_elem_num];
        HGEMM_CHECK(m_host_ptr);
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(half)));
        HGEMM_CHECK(m_dev_ptr);

        // fill everything with zeros
        for (size_t i = 0; i < m_elem_num; ++i) {
            m_host_ptr[i] = __float2half(0);
        }

        // fill with csr values
        for (size_t row = 0; row < m_row; row++)
        {
            for (size_t j = csrRowPtr_host[row]; j < csrRowPtr_host[row + 1]; j++) 
            {
                size_t col = csrColIdx_host[j];
                half val = csrVal_host[j];
                m_host_ptr[row * m_col + col] = val;
            }
        }

        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice));
    }

    void csrToBcsr() {
        // first prepare the info arrays
        size_t numColRegions = (m_col + MMA_K - 1) / MMA_K;
        size_t numRowRegions = (m_row + MMA_M - 1) / MMA_M;

        numberOfBlocks = numRowRegions * numColRegions;
        //printf("numblocks %d\n", numberOfBlocks);
        
        blockInfo_host = (int *) calloc(sizeof(int), numberOfBlocks);
        // 0 - zero block
        // 1 - 2:4 sparse block
        // 2 - dense block
        for (size_t row = 0; row < m_row; row++)
        {
            for (size_t j = csrRowPtr_host[row]; j < csrRowPtr_host[row + 1]; j++) 
            {
                size_t col = csrColIdx_host[j];
                //printf("%f\n", csrValA[j]);
                //printf("col %d\n", col);
                size_t rowRegion = row / MMA_M;
                size_t colRegion = col / MMA_K;
                //printf("row_reg %d  col reg %d \n", rowRegion, colRegion);
                size_t blockIndex = rowRegion * numColRegions + colRegion;
                //printf("block  index %d\n", blockIndex);
                if (blockInfo_host[blockIndex] == 0)  // zero block, stops being 0, becomes sparse
                {
                    blockInfo_host[blockIndex] = 1;
                    nonzeroBlocks += 1;
                    sparseBlocks++;
                }
                else if (blockInfo_host[blockIndex] == 1)  // sparse block
                {
                    // check can it still be sparse
                    // should I check previous two or I am in new part for 2:4
                    size_t relative24Index = col % 4;
                    if (relative24Index == 2 || relative24Index == 3)
                    {
                        if (j >= csrRowPtr_host[row] + 2 && csrColIdx_host[j-1] == col-1 && csrColIdx_host[j-2] == col-2) 
                        {
                            blockInfo_host[blockIndex] = 2;
                            denseBlocks++;
                            sparseBlocks--;
                        }
                    }
                    
                }
            }
        }

        size_t relativeIndex = 0;
        relativeBlockIndexMapping_host = (int*) malloc(numberOfBlocks * sizeof(int));
        for (size_t i = 0; i < numberOfBlocks; i++)
        {
            relativeBlockIndexMapping_host[i] = (blockInfo_host[i] != 0) ? relativeIndex++ : -1;
            //printf("relative [%d] = %d\n", i, relativeBlockIndexMapping[i]);
        }

        // get the bcsr
        bcsrRowPtr_host = (int*)calloc(sizeof(int), (m_row / MMA_M + 1));
        bcsrColIdx_host = (int*)malloc(nonzeroBlocks * sizeof(int));
        bcsrVal_host = (half*)calloc(sizeof(half), nonzeroBlocks * MMA_M * MMA_K);

        size_t num_blocks = 0;
        
        // Do the rowPtrBcsr and colIdxBcsr
        for (size_t row = 0; row < m_row; row += MMA_M) {
            // printf("Problem in 314?\n");
            bcsrRowPtr_host[row / MMA_M] = num_blocks; // Update rowPtr
            //printf("rowPtr[%d] = %d\n", row/MMA_M, num_blocks);

            // Iterate through columns
            for (size_t col = 0; col < m_col; col += MMA_K) {
                size_t current_block = row / MMA_M * numColRegions + col / MMA_K;
                //printf("Problem in 320?");
                if (blockInfo_host[current_block] == 0)
                {
                    continue;
                }
                //printf("Problem in 325?");
                bcsrColIdx_host[num_blocks] = col; // not relative bcsr columns index / MMA_K if want relative
                //printf("colIdx[%d] = %d\n", num_blocks, col);
                num_blocks++;

            }
        }

        //printf("Problem in 372?");
        bcsrRowPtr_host[m_row / MMA_M] = num_blocks; // Update last entry of rowPtr
        //printf("rowPtr[%d] = %d\n", numRows / MMA_M, num_blocks);

            
        //printf("%d total blocks\n", totalNumberOfBlocks);
        
        // Do the valuesBcsr
        for (size_t row = 0; row < m_row; row++)
        {
            for (size_t j = csrRowPtr_host[row]; j < csrRowPtr_host[row + 1]; j++) 
            {
                size_t col = csrColIdx_host[j];
                //printf("col %d\n", col);
                size_t rowRegion = row / MMA_M;
                size_t colRegion = col / MMA_K;
                //printf("row_reg %d  col reg %d \n", rowRegion, colRegion);
                size_t blockIndex = rowRegion * numColRegions + colRegion;
                half val = csrVal_host[j];
                //printf("val %f\n", val);
                size_t offset = row % MMA_M * MMA_K + col % MMA_K;
                size_t bcsrIndex = relativeBlockIndexMapping_host[blockIndex] * MMA_M * MMA_K + offset;
                //printf("relativeIndex %d x %d +  offset %d = %d\n", relativeBlockIndexMapping[blockIndex], blockSize, offset, bcsrIndex);
                bcsrVal_host[bcsrIndex] = val;
            }
        }

        /* for (int i = 0; i < num_blocks * blockSize; i++)
        {
            if (i % blockSize == 0)
            {
                printf("\n");   
            }
            printf("%f\n", *(*valuesBcsr + i));
            
        } */

        // create the data structures for locations of nnz blocks
        // not needed for now

    }

    void csrToBcsrKnapsacking() {
        size_t numColRegions = (m_col + MMA_K - 1) / MMA_K;
        size_t numRowRegions = (m_row + MMA_M - 1) / MMA_M;
        std::vector<std::vector<std::tuple<size_t, size_t, float, long, size_t>>> startsEndsOfCols(numRowRegions);

        bcsrRowPtr_host = (int*)calloc(sizeof(int), (m_row / MMA_M + 1));
        std::vector<size_t> vecOfColIdx;
        for (size_t row = 0; row < m_row; row += MMA_M) {
            bcsrRowPtr_host[row / MMA_M] = nonzeroBlocks;
            //printf("[%lu] = %d\n", (unsigned long) row / MMA_M, nonzeroBlocks);
            std::vector<size_t> columnsInBlock;
            for (size_t pointer = row; pointer < row + MMA_M; pointer++) {
                // dodaj iteraciju po columnima
                for (size_t j = csrRowPtr_host[pointer]; j < csrRowPtr_host[pointer+1]; j++) {
                    // columnsInBlock.push_back(csrColIdx_host[j]);
                    startsEndsOfCols[row / MMA_M].push_back(std::make_tuple(csrColIdx_host[j], pointer, __half2float(csrVal_host[j]), -1, 0));
                }
            }
            //std::sort(columnsInBlock.begin(), columnsInBlock.end());
            std::sort(startsEndsOfCols[row / MMA_M].begin(), startsEndsOfCols[row / MMA_M].end(),
            [](const std::tuple<size_t, size_t, float, long, size_t>& tuple1, const std::tuple<size_t, size_t, float, long, size_t>& tuple2) {
                  return std::get<0>(tuple1) < std::get<0>(tuple2);
              });

            for (int i = 0; i < startsEndsOfCols[row / MMA_M].size(); i++) {
                auto a = startsEndsOfCols[row / MMA_M][i];
                //printf("%lu %lu %f %ld %lu\n", std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a), std::get<4>(a));
            }

            //if (columnsInBlock.empty()) {
            if (startsEndsOfCols[row / MMA_M].empty()) {
                continue;
            }

            /* size_t start = columnsInBlock[0];
            size_t end = start + MMA_K;
            startsEndsOfCols[row / MMA_M].push_back(start);
            startsEndsOfCols[row / MMA_M].push_back(end);
            nonzeroBlocks++;
            for (size_t i = 1; i < columnsInBlock.size(); ++i) {
                if (columnsInBlock[i] >= end) {
                    start = columnsInBlock[i];
                    end = start + MMA_K;
                    startsEndsOfCols[row / MMA_M].push_back(start);
                    startsEndsOfCols[row / MMA_M].push_back(end);
                    nonzeroBlocks++;
                }
            } */

            size_t firstColumn = std::get<0>(startsEndsOfCols[row / MMA_M][0]);
            size_t lastColumn = std::get<0>(startsEndsOfCols[row / MMA_M].back());
            size_t start;
            size_t span = lastColumn - firstColumn + 1;
            size_t potentialAddLeft = firstColumn - 0;
            size_t potentialAddRight = m_col - lastColumn - 1;
            size_t to_add = MMA_K - span % MMA_K;

            if (span % MMA_K == 0) {
                start = std::get<0>(startsEndsOfCols[row / MMA_M][0]);
            }
            else {
                
                if (potentialAddRight >= to_add) {
                    start = std::get<0>(startsEndsOfCols[row / MMA_M][0]);
                }
                else if (potentialAddLeft >= to_add) {
                    start = std::get<0>(startsEndsOfCols[row / MMA_M][0]) - to_add;
                }
                else {
                    start = std::get<0>(startsEndsOfCols[row / MMA_M][0]) - potentialAddLeft;
                }
            }
             
            vecOfColIdx.push_back(start);
            size_t end = start + MMA_K;
            std::get<3>(startsEndsOfCols[row / MMA_M][0]) = nonzeroBlocks;
            // change relative column
            std::get<4>(startsEndsOfCols[row / MMA_M][0]) = std::get<0>(startsEndsOfCols[row / MMA_M][0]) - start;
            //nonzeroBlocks++;
            /* auto a = startsEndsOfCols[row / MMA_M][0];
            printf("%lu %lu %f %ld    ==>\n", std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a));
            printf("span %lu addLeft %lu addRight %lu toAdd %lu ", span, potentialAddLeft, potentialAddRight, to_add);
            printf(" start  %lu end %lu \n", start, end); */
            for (size_t i = 1; i < startsEndsOfCols[row / MMA_M].size(); ++i) {
                //a = startsEndsOfCols[row / MMA_M][i];
                //printf("%lu %lu %f %ld    ==>\n", std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a));
                if (std::get<0>(startsEndsOfCols[row / MMA_M][i]) >= end) {
                    // printf("tusam\n");
                    span = lastColumn - std::get<0>(startsEndsOfCols[row / MMA_M][i]) + 1;
                    potentialAddLeft = std::get<0>(startsEndsOfCols[row / MMA_M][i]) - end;
                    to_add = MMA_K - span % MMA_K;
                    //printf("span %lu addLeft %lu addRight %lu toAdd %lu ", span, potentialAddLeft, potentialAddRight, to_add);
                    if (span % MMA_K == 0) {
                        start = std::get<0>(startsEndsOfCols[row / MMA_M][i]);
                    }
                    else {
                        if (potentialAddRight >= to_add) {
                            start = std::get<0>(startsEndsOfCols[row / MMA_M][i]);
                        }
                        else if (potentialAddLeft >= to_add) {
                            start = std::get<0>(startsEndsOfCols[row / MMA_M][i]) - to_add;
                        }
                        else {
                            start = std::get<0>(startsEndsOfCols[row / MMA_M][i]) - potentialAddLeft;
                        }
                    }
                    
                    vecOfColIdx.push_back(start);
                    end = start + MMA_K;
                    //printf(" start  %lu end %lu \n", start, end);
                    nonzeroBlocks++;
                }
                
                //printf("cols %lu %lu %lu\n", (unsigned long) start, (unsigned long) end, (unsigned long) nonzeroBlocks);
                std::get<3>(startsEndsOfCols[row / MMA_M][i]) = nonzeroBlocks;
                std::get<4>(startsEndsOfCols[row / MMA_M][i]) = std::get<0>(startsEndsOfCols[row / MMA_M][i]) - start;
            }
            nonzeroBlocks++;
        }
        
        bcsrRowPtr_host[m_row / MMA_M] = nonzeroBlocks;

        //printf ("%lu nnzblocks\n", (unsigned long) nonzeroBlocks);
        bcsrColIdx_host = (int*)malloc(nonzeroBlocks * sizeof(int));
        bcsrVal_host = (half*)calloc(sizeof(half), nonzeroBlocks * MMA_M * MMA_K);

        size_t current_idx = 0;
        for (size_t i = 0; i < nonzeroBlocks; i++) {
            bcsrColIdx_host[i] = vecOfColIdx[i];
            //printf("%d ", bcsrColIdx_host[i]);
        }
        

        //printf("\n");
        for (size_t rowRegion = 0; rowRegion < startsEndsOfCols.size(); rowRegion++) {
            for (size_t element = 0; element < startsEndsOfCols[rowRegion].size(); element++) {
                    size_t col = std::get<0>(startsEndsOfCols[rowRegion][element]);
                    size_t row = std::get<1>(startsEndsOfCols[rowRegion][element]);
                    half val = __float2half(std::get<2>(startsEndsOfCols[rowRegion][element]));
                    size_t relBlock = std::get<3>(startsEndsOfCols[rowRegion][element]);
                    size_t relColumn = std::get<4>(startsEndsOfCols[rowRegion][element]);  
                    bcsrVal_host[relBlock * MMA_M * MMA_K + (row % MMA_M) * MMA_K + relColumn] = val;
                    //printf("%lu %lu %f %ld %lu ==> %lu\n", row, col, __half2float(val), relBlock, relColumn, relBlock * MMA_M * MMA_K + (row % MMA_M) * MMA_K + relColumn);               
                }
                //printf("\n");
        }

        /* printf("\nrowptr\n");
        //printf("%d rowptr size", (int)m_row / MMA_M + 1);
        for (int i = 0; i < m_row / MMA_M + 1; i++) {
            printf("%d ", bcsrRowPtr_host[i]);
        }
        printf("\n");

        for (int i = 0; i < nonzeroBlocks * MMA_M * MMA_K; i++) {
            
            if (i % (MMA_K * MMA_M) == 0) {
                printf ("\n\n");
            }
            else if (i % MMA_K == 0) {
                printf("\n");

            }
            printf("%f ", __half2float(bcsrVal_host[i]));
        } */

        
    }

private:
    size_t m_row = 0;
    size_t m_col = 0;
    std::string m_name = "Matrix";
    // the threshold of the random matrix will affect the difference of the hgemm results
    float m_min = -1.0;
    float m_max = 1.0;

    size_t m_elem_num = 0;
    half *m_host_ptr = nullptr;
    half *m_dev_ptr = nullptr;

    double m_max_diff = 0.0;
    double m_avg_diff = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(SparseMatrix);

    char* filename = "";

    size_t nnz = 0;

    half *csrVal_host = nullptr;
    int *csrRowPtr_host = nullptr;
    int *csrColIdx_host = nullptr;

    half *bcsrVal_host = nullptr;
    int *bcsrRowPtr_host = nullptr;
    int *bcsrColIdx_host = nullptr; 


    half *csrVal_dev = nullptr;
    int *csrRowPtr_dev = nullptr;
    int *csrColIdx_dev = nullptr;

    half *bcsrVal_dev = nullptr;
    int *bcsrRowPtr_dev = nullptr;
    int *bcsrColIdx_dev = nullptr;

    int *blockInfo_host = nullptr;
    int *relativeBlockIndexMapping_host = nullptr;

    int *blockInfo_dev = nullptr;
    int *relativeBlockIndexMapping_dev = nullptr;

    size_t numberOfBlocks = 0;
    size_t nonzeroBlocks = 0;
    size_t sparseBlocks = 0;
    size_t denseBlocks = 0;
};
