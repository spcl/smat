#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <random> //std::shuffle
#include <stdexcept>

#include <algorithm>    // std::sort
#include <numeric> //std::iota
#include <iterator>
#include "matrices.h"
#include "utilities.h"

using namespace std;

void CSR::clean()
{
    /*----------------------------------------------------------------------
    | Free up memory allocated for CSR structs.
    |--------------------------------------------------------------------*/

    if (rows + cols <= 1) return;

    if (ma && !pattern_only) 
    {
        for(intT i = 0; i < rows; i++)
        {
            if (ma[i]) delete[] ma[i];
        }
    }

    if (ja) 
    {
        for(intT i = 0; i < rows; i++)
        {
            if (ja[i]) delete[] ja[i];
        }
    }
    
    if (nzcount) delete[] nzcount;
    
    rows = 0;
    cols = 0; 
}


void CSR::multiply(DataT* mat_B, intT B_cols, DataT_C* mat_C)
{
    //A*B, A is CSR, B and C column-wise storage

    for(intT i = 0; i < rows; i++)
    {
            for (intT nz = 0; nz < nzcount[i]; nz++)
            {
                intT col = ja[i][nz];
                DataT val = pattern_only?1:ma[i][nz];
                for(intT j = 0; j < B_cols; j++)
                {
                    mat_C[i + j*rows] += val*mat_B[col + j*rows];
                }
            }
    }
}

void CSR::permute_rows(vector<intT> permutation)
//permute the rows of the matrix according to "permutation"
{   
    if (permutation.size() != rows)
        throw std::invalid_argument("CSR.permute_rows argument bust have same lenght as rows");

    permute(ja,permutation);
    if (!pattern_only) permute(ma, permutation);
    permute(nzcount, permutation);
}

void CSR::permute_cols(vector<intT> permutation)
//permute the rows of the matrix according to "permutation"
{   
    if (permutation.size() != cols)
        throw std::invalid_argument("CSR.permute_rows argument bust have same lenght as rows");


    for (intT i = 0; i < rows; i++)
    {
        vector<intT> tmp_idxs;
        for (intT nz = 0; nz < nzcount[i]; nz++)
        {
            intT permuted_idx = permutation[ja[i][nz]];
            tmp_idxs.push_back(permuted_idx);
        }
        sort(tmp_idxs.begin(),tmp_idxs.end());
        copy(tmp_idxs.begin(), tmp_idxs.end(),ja[i]);
        if (!pattern_only) cout << "WARNING: column reordering not implemented for weighted matrices" << endl;
    }
}

void CSR::reorder(vector<intT> grouping)
//permute the rows so that row i and row j are adjacent if grouping[i] == grouping[j]
{
    if (grouping.size() != rows)
        throw std::invalid_argument("CSR.reorder argument bust have same lenght as rows");

    vector<intT> v = get_permutation(grouping);
    permute_rows(v);
}

void CSR::reorder_by_degree(bool descending)
{
    auto asc_comparator = [&](int i, int j)
    {
        return nzcount[i] < nzcount[j];
    };

    auto desc_comparator = [&](int i, int j)
    {
        return nzcount[i] >= nzcount[j];
    };

    vector<intT> v(rows);
    iota(v.begin(), v.end(), 0);
    
    for (intT i = 0; i < rows; i++)
    {
        try
        {
            intT test = nzcount[i];
        }
        catch(const std::exception& e)
        {
            std::cout << "REORDERING FAILED ON ROW " << i <<  e.what() << '\n';
        }
        
    }

    if (descending) sort (v.begin(), v.end(), desc_comparator);
    else sort (v.begin(), v.end(), asc_comparator);
    permute_rows(v);

}

void CSR::scramble()
{
    //randomly permute rows (TODO better randomness)
    vector<intT> v(rows);
    iota(v.begin(), v.end(), 0);

    random_shuffle(v.begin(),v.end());

    permute_rows(v);
}


void CSR::save_to_edgelist(std::ofstream& outfile, std::string delimiter, bool pattern_only, MatrixFormat mat_fmt)
{
    for (int i = 0; i < rows; i++)
    {
        for (int nz = 0; nz < nzcount[i]; nz++)
        {
            if (mat_fmt == mtx) outfile << ja[i][nz] << delimiter << i << "\n";
            else outfile << i << delimiter << ja[i][nz] << "\n";
        }
    }
}


void CSR::read_from_edgelist(ifstream& infile, string delimiter, bool pattern_only, MatrixFormat mat_fmt, bool symmetrize)
//reads edgelist into the CSR.
{
    this->pattern_only = pattern_only;

    intT last_node = -1;
    intT current_node;
    string temp;
    vector<vector<intT>> pos_holder;
    vector<vector<DataT>> val_holder;
    intT max_column = 0;
    intT i = -1; 
    DataT val;
    intT total_nonzeros = 0;

    bool triangular = true;

    while (infile.peek() == '#' or infile.peek() == '%') infile.ignore(2048, '\n');
    if (mat_fmt == mtx) infile.ignore(2048, '\n');
    while (getline(infile, temp)) {

        total_nonzeros++;
        int del_pos = temp.find(delimiter);
        int del_size = delimiter.length();

        string first_node_string = temp.substr(0, del_pos); //retrieve the part of the string before the delimiter
        current_node = stoi(first_node_string);
        temp.erase(0, del_pos + del_size);
        
        del_pos = temp.find(delimiter);
        string second_node_string = temp.substr(0, del_pos); //retrieve the part of the string after the delimiter
        intT child = stoi(second_node_string);

        if (mat_fmt == mtx)
        {
            swap(current_node, child);
        }

        if (child < current_node) triangular = false;

        max_column = std::max(max_column, child);


        if (not pattern_only)
        {
            temp.erase(0, del_pos + del_size);
            del_pos = temp.find(delimiter);
            string val_string = temp.substr(0, del_pos); //retrieve the part of the string after the delimiter
            val = stof(val_string);
        }


        //fill with empty lines to reach current row
        if (current_node > i)
        {
            while (i < current_node)
            {
                vector<intT> new_pos_row;
		        pos_holder.push_back(new_pos_row);
                if (not pattern_only)
                {
                    vector<DataT> new_val_row;
                    val_holder.push_back(new_val_row);
                }
                i++;
            }
        }
        else if (current_node < i)
            throw std::invalid_argument("CSR.read_from_edgelist indices must be in ascending order");

        pos_holder[i].push_back(child);
    	if (not pattern_only) val_holder[i].push_back(val);
    }


    if (symmetrize && triangular)
    {
        for (intT i = 0; i < pos_holder.size(); i++)
        {
            for (intT nz = 0; nz < pos_holder[i].size(); nz++)
            {
                intT j = pos_holder[i][nz];
                auto it = lower_bound(pos_holder[j].begin(), pos_holder[j].end(), i);
                if (it == pos_holder[j].end() || *it != i) {
                    pos_holder[j].insert(it, i);
                    if (not pattern_only) 
                    {
                        throw std::invalid_argument("Invalid option: -e 1; symmetrize only implemented for unweighted graphs");
                    }
                }
            }

        }
    }


    rows = pos_holder.size();
    cols = max_column + 1;
    nzcount = new intT[rows];
    ja = new intT*[rows];
    if (not pattern_only) 
    {
        ma = new DataT*[rows];
    }

    for (intT i = 0; i < pos_holder.size(); i++)
    {
        auto row_pos = pos_holder[i];
        nzcount[i] = row_pos.size();
        ja[i] = new intT[nzcount[i]]; 
        std::copy(row_pos.begin(), row_pos.end(), ja[i]);
	    pos_holder[i].clear();
	    
        if (not pattern_only)
        {
            auto row_val = val_holder[i];
            ma[i] = new DataT[nzcount[i]];
            std::copy(row_val.begin(), row_val.end(), ma[i]);
            val_holder[i].clear();
        }
    }

    pos_holder.clear();
    val_holder.clear();
}

void CSR::print(intT verbose)
{
    cout << "PRINTING A CSR MATRIX (arrays only)" << endl;
    cout << "ROWS: " << rows << " COLS: " << cols << " PATTERN_only: " << pattern_only << endl; 
    cout << "NZ: " << nztot() << endl;

    if (verbose > 1)
    {
        cout << "JA:" << endl;
        for (intT i = 0; i < rows; i++)
        {
            cout << "-- ";
            for (intT j = 0; j < nzcount[i]; j++)
            {
                cout << ja[i][j] << " ";
            }
            cout << endl;

        }

        if (!pattern_only)
        {        
            cout << "MA:" << endl;
            for (intT i = 0; i < rows; i++)
            {
                cout << "-- ";
                for (intT j = 0; j < nzcount[i]; j++)
                {
                    cout << ma[i][j] << " ";
                }
                cout << endl;
            }
        }

        cout << "NZCOUNT:" << endl;
        for (intT i = 0; i < rows; i++)
        {
                cout << nzcount[i] << " ";
        }
        cout << endl;
    }

    if (verbose > 0)
        {
            //loop through rows
            for (intT i = 0; i < rows; i++)
            {
                intT j = 0;
                for (intT nzs = 0; nzs < nzcount[i]; nzs++) 
                {
                    intT nz_column = ja[i][nzs]; //find column (row) index of next nonzero element
                    
                    DataT elem;
                    if (!pattern_only) elem = ma[i][nzs]; //value of that element;
                    else elem = 1;
                
                    while (j < nz_column)
                    {
                        j++;
                        cout << 0 << " ";
                    }
                    cout << elem << " ";
                    j++;
                }
                while (j < cols)
                {
                    j++;
                    cout << 0 << " ";
                }

            cout << endl;
            }
        }

}
