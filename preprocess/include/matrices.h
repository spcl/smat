#pragma once


#include <string>
#include <vector>
#include <fstream>
#include "input.h"
#include "definitions.h"

struct CSR
{
    /*--------------------------------------------------------------
    | Compressed sparse row (CSR) matrix format,
    | generally not square
    |==========
    |       cmat  = a CSR struct
    |       rows  = # of columns of the matrix
    |       cols  = # of columns of the matrix
            pattern_only   = 1: sparsity pattern only (boolean matrix)
    |               1: data and pattern
    |--------------------------------------------------------------------   */
    intT rows;      /* number of rows                                        */
    intT cols;      /* number of cols                                        */
    intT* nzcount;  /* number of nonzero entry in each row (column)          */
    intT** ja;      /* pointer-to-pointer to store column (row) indices      */
    DataT** ma;   /* pointer-to-pointer to store nonzero entries           */

    bool pattern_only; // 1 if the matrix is patter only; 0 otherwise

    void clean();
    void read_from_edgelist(std::ifstream& infile, std::string delimiter = " ", bool pattern_only = true, MatrixFormat mat_fmt = mtx, bool symmetrize = false);
    void save_to_edgelist(std::ofstream& outfile, std::string delimiter = " ", bool pattern_only = true, MatrixFormat mat_fmt = mtx);

    void reorder(std::vector<intT> grouping);
    void reorder_by_degree(bool descending = true);
    void permute_rows(std::vector<intT> permutation);
    void permute_cols(std::vector<intT> permutation);
    void scramble();

    void multiply(DataT* mat_B, intT B_cols, DataT_C* mat_C);
    void print(intT verbose = 0);
    intT nztot()
    {
        intT nztot = 0;
        for (intT i = 0; i < rows; i++)
        {
            nztot += nzcount[i];
        }
        return nztot;
    }


    //constructor for edgelist data
    CSR(std::ifstream& infile, std::string delimiter = " ", bool pattern_only = true, MatrixFormat mat_fmt = mtx)
    {
        read_from_edgelist(infile, delimiter, pattern_only, mat_fmt);
    }


    //constructor from command line object
    CSR(CLineReader &cli)
    {
        std::ifstream fin;
        fin.open(cli.filename_);
        read_from_edgelist(fin, cli.reader_delimiter_, cli.pattern_only_, (MatrixFormat) cli.mat_fmt_, cli.symmetrize_);
        switch(cli.reorder_)
        {
            case 1:
                reorder_by_degree(true);
                break;
            case -1:
                reorder_by_degree(false);
                break;
            case 2:
                scramble();
                break;
        }
    }
    
    
    //destructor cleans all arrays
    ~CSR()
    {
        clean();
    }

};

struct VBR
{
    intT rows;
    intT cols;
    intT block_rows;  	            /* the block row dimension of the matrix    	        */
    intT block_cols;	            /* the block column dimension of the matrix   	        */
    intT* nzcount;	        /* i-th elements is the number of nonzero blocks in block-row i*/
    intT* jab;              /* block-columns indices of nonzero blocks      */
    intT* row_part;         /*row_part[i] is the index of the first row of the i-th block. Last element is number of rows for convenience  */
    DataT* mab;             /* array containing all entries, block by block. Blocks are stored in row-major order, while entries in the blocks are stored in column-major format.        */
    intT block_col_size;                  
    intT nztot;              /* total number of nonzero elements in mab*/


    //destructor cleans all arrays
    ~VBR()
    {
        clean();
    }

    void clean();
    void print(int verbose = 0);

    int partition_check(const std::vector<intT> &candidate_part);
    void fill_from_CSR(const CSR& cmat,const std::vector<intT> &row_partition, intT block_size);
    void fill_from_CSR_inplace(const CSR& cmat,const std::vector<intT> &grouping, intT col_block_size, intT row_block_size = 0, bool force_fixed_size = false);
    void fill_from_CSR_inplace(const CSR& cmat, intT row_block_size, intT col_block_size, bool force_fixed_size = false);
    DataT* get_block_start(intT row_block_idx);
    void multiply(DataT* B, int B_cols, DataT_C* C);
};


void serial_batched(const VBR& vbmatA, DataT* d_B, int B_rows, DataT_C* d_C);
