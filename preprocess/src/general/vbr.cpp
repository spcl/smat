#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>    // std::sort, std::count
#include <numeric> //std::iota
#include "matrices.h"
#include "utilities.h"
#include <cstring>

using namespace std;

void VBR::clean()
{
    if (rows + cols <= 1) return;
    
    if (mab) delete[] mab;
    
    if (jab) delete[] jab;

    if (nzcount) delete[] nzcount;

    if (row_part) delete[] row_part;

    rows = 0;
    cols = 0;
    block_rows = 0;
    block_cols = 0;
    nztot = 0;
    block_col_size = 0;
}

DataT* VBR::get_block_start(intT row_block_idx)
{
    //returns the position in the mab vector where entries from a row block are stored. 
    DataT* ptr = mab;
    if (row_block_idx >= block_rows) 
    {
        cerr << " [RANGE ERROR] VBR::get_block_start" << endl;
        return mab;
    }

    for (intT ib = 0; ib < row_block_idx; ib++)
    {
        ptr += (row_part[ib+1] - row_part[ib])*nzcount[ib]*block_col_size;
    }
    return ptr;

}

void VBR::print(int verbose)
{
    cout << "VBR: " << endl;
    cout << "--- rows, cols: " << rows << "," << cols << endl;
    cout << "--- block rows, block cols: " << block_rows << "," << block_cols << endl;
    cout << "--- block columns size: " << block_col_size << endl;
    cout << "--- total nonzero area: " << nztot << endl;
    if (verbose > 0)
    {
        intT* jab_ptr = jab;
        for (intT ib = 0; ib < block_rows; ib++)
        //iterate through block-rows
        {   
            intT row_block_size = row_part[ib+1] - row_part[ib];
            DataT* data_pointer = get_block_start(ib);

            for (intT i = 0; i < row_block_size; i++)
            //iterate through rows in the row-block
            {
                intT jb = 0;
                for (intT nzb = 0; nzb < nzcount[ib]; nzb++)
                //iterate through nonzero blocks in the row-block
                {
                    intT nz_jb = jab_ptr[nzb]; //position of nonzero block
                    while (jb < nz_jb)
                    {
                        for (intT j = 0; j < block_col_size; j++) cout << "0 ";
                        cout << "| ";
                        jb++;
                    }
                    jb++;
                    //print mab  slice
                    for (intT j = 0; j < block_col_size; j++) 
                    {
                        DataT d = data_pointer[nzb*block_col_size*row_block_size + j*row_block_size + i];
                        cout << d << " " ;
                    }    
                    cout << "| ";
                }

                while (jb < block_cols && jb*block_col_size < cols)
                {
                    for (intT j = 0; j < block_col_size; j++) cout << "0 ";
                    {
                        cout << "| ";
                    }
                    jb++;
                }
                cout << endl;
            }
            cout << endl;
            jab_ptr += nzcount[ib]; //move jab ptr to start of next block_row
        }
        cout << endl;
    }
}

int VBR::partition_check(const vector<intT> &candidate_part)
{
//checks that a vector is a valid partition; anything different than 0 is an error;
    if (candidate_part.size() == 0) return 1;
    if (candidate_part.back() != rows) return 2;
    for (intT i = 1; i < candidate_part.size(); i++) 
    {
        if (candidate_part[i] < candidate_part[i - 1]) return 3;
    }
    return 0;
}


void VBR::fill_from_CSR_inplace(const CSR& cmat,intT row_block_size, intT col_block_size, bool force_fixed_size)
{

    //generate fixed_size grouping
    vector<intT> grouping;
    for (intT i = 0; i < cmat.rows; i++)
    {
        grouping.push_back(i/row_block_size);    
    }
    //fill from fixed_size grouping
    fill_from_CSR_inplace(cmat, grouping, col_block_size, row_block_size, force_fixed_size);
}


void VBR::fill_from_CSR_inplace(const CSR& cmat,const vector<intT> &grouping, intT col_block_size, intT row_block_size, bool force_fixed_size)
{
    //fill the VBR with entries from a CSR, with rows permuted and grouped according to grouping.

    vector<intT> row_partition = get_partition(grouping);
    vector<intT> row_permutation = get_permutation(grouping);


    if (force_fixed_size){
        rows = ((cmat.rows -1)/row_block_size + 1)*row_block_size;
        cols = ((cmat.cols -1)/col_block_size + 1)*col_block_size;
        row_partition[row_partition.size() - 1] = rows;
        for (int i = row_permutation.size(); i < rows; i++) row_permutation.push_back(i);
    }
    else{
        rows = cmat.rows;
        cols = cmat.cols;
    }


    block_col_size = col_block_size;
    block_cols = (cols - 1)/col_block_size + 1;
    block_rows = row_partition.size() - 1;

    //partition check

    if (int a = partition_check(row_partition) != 0)
    {
        cerr << "PARTITION CHECK ERROR: partition check failed with error " << a << endl;

    }
    row_part = new intT[row_partition.size()];
    copy(row_partition.begin(),row_partition.end(), row_part);

    nzcount = new intT[block_rows]{0};

    vector<DataT> mab_vec;
    vector<intT> jab_vec;

    //copy data block_row by block_row
    for(intT ib = 0; ib < block_rows; ib++)
    {
        vector<bool> nonzero_flags(block_cols, false);
        intT row_block_size = row_part[ib+1] - row_part[ib];

        //flag nonzero blocks
        for (intT i_reordered = row_part[ib]; i_reordered < row_part[ib+1]; i_reordered++)
        {
            intT i = row_permutation[i_reordered];

            if (i >= cmat.rows)     
                continue;
            for (intT nz = 0; nz < cmat.nzcount[i]; nz++)
            {
                intT j = cmat.ja[i][nz];
                nonzero_flags[j/block_col_size] = true;
            }
        }

        //fill jab vector
        for (intT jb = 0; jb < block_cols; jb++)
        {
            if (nonzero_flags[jb]) jab_vec.push_back(jb);
        }

        //fill nzcount
        nzcount[ib] = std::count(nonzero_flags.begin(), nonzero_flags.end(), true);


        //fill mab vector
        intT current_mab_size = mab_vec.size(); 
        mab_vec.resize(current_mab_size + nzcount[ib]*row_block_size*block_col_size, 0);
        for (intT i_reordered = row_part[ib]; i_reordered < row_part[ib+1]; i_reordered++)
        {
            intT i = row_permutation[i_reordered];

            if (i >= cmat.rows)
                continue;
            for (intT nz = 0; nz < cmat.nzcount[i]; nz++)
            {
                intT j = cmat.ja[i][nz];
                DataT d;
                if (cmat.pattern_only) d = 1;
                else d = cmat.ma[i][nz];

                //find position of d in mat
                intT j_block_position = j/col_block_size;
                intT tmp_block_count = std::count(nonzero_flags.begin(), nonzero_flags.begin() + j_block_position, true); //how many nz_blocks before current
                //intT tmp_mab_pos = current_mab_size + tmp_block_count*block_col_size*row_block_size + (i_reordered - row_part[ib])*block_col_size + j%block_size; 
                intT tmp_mab_pos = current_mab_size + tmp_block_count*block_col_size*row_block_size + row_block_size*(j%col_block_size) + (i_reordered - row_part[ib]); //column-major format
             
                mab_vec[tmp_mab_pos] = d;
            }
        }

    }

    nztot = mab_vec.size();
    jab = new intT[jab_vec.size()];
    mab = new DataT[mab_vec.size()];
    copy(jab_vec.begin(), jab_vec.end(),jab);
    copy(mab_vec.begin(),mab_vec.end(), mab);
}

void VBR::fill_from_CSR(const CSR& cmat,const vector<intT> &row_partition, intT block_size)
{
    //fill the VBR with entries from a CSR, with rows appearing in the same order and divided according to row_partition.

    rows = cmat.rows;
    cols = cmat.cols;
    block_col_size = block_size;
    
    block_cols = (cols - 1)/block_size + 1;
    block_rows = row_partition.size() - 1;


    //partition check

    if (int a = partition_check(row_partition) != 0)
    {
        cerr << "PARTITION CHECK ERROR: partition check failed with error " << a << endl;

    }
    row_part = new intT[row_partition.size()];
    copy(row_partition.begin(),row_partition.end(), row_part);

    nzcount = new intT[block_rows]{0};

    vector<DataT> mab_vec;
    vector<intT> jab_vec;

    //copy data block_row by block_row
    for(intT ib = 0; ib < block_rows; ib++)
    {
        vector<bool> nonzero_flags(block_cols, false);
        intT row_block_size = row_part[ib+1] - row_part[ib];

        //flag nonzero blocks
        for (intT i = row_part[ib]; i < row_part[ib+1]; i++)
        {
            for (intT nz = 0; nz < cmat.nzcount[i]; nz++)
            {
                intT j = cmat.ja[i][nz];
                nonzero_flags[j/block_col_size] = true;
            }
        }

        //fill jab vector
        for (intT jb = 0; jb < block_cols; jb++)
        {
            if (nonzero_flags[jb]) jab_vec.push_back(jb);
        }

        //fill nzcount
        nzcount[ib] = std::count(nonzero_flags.begin(), nonzero_flags.end(), true);


        //fill mab vector
        intT current_mab_size = mab_vec.size(); 
        mab_vec.resize(current_mab_size + nzcount[ib]*row_block_size*block_col_size, 0);
        for (intT i = row_part[ib]; i < row_part[ib+1]; i++)
        {
            for (intT nz = 0; nz < cmat.nzcount[i]; nz++)
            {
                intT j = cmat.ja[i][nz];
                DataT d;
                if (cmat.pattern_only) d = 1;
                else d = cmat.ma[i][nz];

                //find position of d in mat
                intT j_block_position = j/block_size;
                intT tmp_block_count = std::count(nonzero_flags.begin(), nonzero_flags.begin() + j_block_position, true); //how many nz_blocks before current
                //intT tmp_mab_pos = current_mab_size + tmp_block_count*block_col_size*row_block_size + (i - row_part[ib])*block_col_size + j%block_size; //row-major format
                intT tmp_mab_pos = current_mab_size + tmp_block_count*block_col_size*row_block_size + row_block_size*(j%block_size) + (i - row_part[ib]); //column-major format

                mab_vec[tmp_mab_pos] = d;
            }
        }

    }

    nztot = mab_vec.size();
    jab = new intT[jab_vec.size()];
    mab = new DataT[mab_vec.size()];
    copy(jab_vec.begin(), jab_vec.end(),jab);
    copy(mab_vec.begin(),mab_vec.end(), mab);
}

void VBR::multiply(DataT* B, int B_cols, DataT_C* C)
{
//multiplies a VBS matrix (vbmatA) and dense matrix (B); stores A*B into (C)
    //vbmatA:       column-major entries (in-block) storage;
    //              row-major block storage; 
    //B:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    //C:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    
    int B_rows = cols;
    int C_rows = rows;
    int C_cols = B_cols;

    intT mat_idx = 0; //keeps writing position for mat
    intT vbmat_idx = 0; //keeps reading position for vbmat 
    intT ja_count = 0; //keeps total nonzero blocks count;
    intT rows_in_block;
    intT* jab_loc = jab;

    //loop through all blocks
    for(intT ib = 0; ib < block_rows; ib++ )      //loop horizontally through block rows
    {
        rows_in_block = row_part[ib + 1] - row_part[ib]; //the row height of the block
        
        for(intT nzs = 0; nzs < nzcount[ib]; nzs++)        //loop horizontally through nonzero blocks

        {
            intT jb = *jab_loc;             //the block row position of a nonzero block 

            auto d_B_block = B + block_col_size*jb;    //access the vertical block of B that is going to be multiplied with blocks of A in block-row ib

            //define the sub-matrices
	          auto d_A_block = mab + vbmat_idx;           //access the block on d_A.
            auto d_C_block = C + row_part[ib];      //access the block on d_C.            
            
            //multiply the dense blocks, accumulate result on C
            for(intT i = 0; i < rows_in_block; i++)
              for(intT j = 0; j < B_cols; j++)
                for(intT k = 0; k < block_col_size; k++)
                  {
                    d_C_block[i + C_rows*j] += d_A_block[i + k*rows_in_block]*d_B_block[k + j*B_rows];
                  }

            //move mab and jab pointers forward
            vbmat_idx += rows_in_block*block_col_size;
            jab_loc++;
	    }

    }

}

void serial_batched(const VBR& vbmatA, DataT* d_B, int B_cols, DataT_C* d_C)
{
//multiplies a VBS matrix (vbmatA) and dense matrix (B); stores B*A into (C)
    //vbmatA:       column-major entries (in-block) storage;
    //              row-major block storage; 
    //B:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    //C:            column-major storage; TODO: allow general storage format (implement through cublas transpose)

    intT A_rows = vbmatA.rows;
    intT A_cols = vbmatA.cols;
    intT B_rows = A_cols;
    intT C_rows = A_rows;
    intT C_cols = B_cols;

    const DataT_C alpha = 1;
    const DataT_C beta = 1;

    intT max_blocks_in_row = 0;
    intT tot_nz_blocks = 0;
    std::vector<intT*> jab_positions;
    std::vector<DataT*> mab_positions;
    intT* current_jab = vbmatA.jab;
    DataT* current_mab = vbmatA.mab;

    for (intT ib = 0; ib < vbmatA.block_rows; ib++)
    {
        jab_positions.push_back(current_jab);
        mab_positions.push_back(current_mab);
        current_jab += vbmatA.nzcount[ib];
        current_mab += vbmatA.nzcount[ib]*vbmatA.block_col_size*(vbmatA.row_part[ib+1] - vbmatA.row_part[ib]);
        max_blocks_in_row = std::max(max_blocks_in_row, vbmatA.nzcount[ib]);
        tot_nz_blocks+= vbmatA.nzcount[ib];
    }

    intT row_block_size = vbmatA.row_part[1] - vbmatA.row_part[0];
    intT block_area = row_block_size*vbmatA.block_col_size;
    std::vector<DataT*> h_d_A;
    std::vector<DataT*> h_d_B;
    std::vector<DataT*> h_d_C;
    DataT* d_A = vbmatA.mab;

    for (intT nzs = 0; nzs < max_blocks_in_row; nzs ++)
    {
        h_d_A.clear();
        h_d_B.clear();
        h_d_C.clear();

        for (intT ib = 0; ib < vbmatA.block_rows; ib++)
        {
            if (nzs >= vbmatA.nzcount[ib]) continue;
            intT jb = *(jab_positions[ib] + nzs);
            cout << "jb: " << jb;
            h_d_A.push_back(mab_positions[ib] + nzs*block_area);
            h_d_B.push_back(d_B + vbmatA.block_col_size*jb);    //access the vertical block of B that is going to be multiplied with blocks of A in block-row ib
            h_d_C.push_back(d_C + row_block_size*C_rows*ib);     //access the block on d_C.
        }

        cout << "*************************NZS: " << nzs << endl;
        cout << "HDA: len " << h_d_A.size() << endl;
        for (auto ptr : h_d_A)
            cout << (ptr)[0] << " "; 
        cout << endl;

        cout << "HDB: len " << h_d_B.size() << endl;
        for (auto ptr : h_d_B)
            cout << ptr[0] << "( " << d_B[5] << " )" <<endl; 
        cout << endl;


    }

}
