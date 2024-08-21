#include <vector>
#include "utilities.h"
#include <numeric> //std::iota
#include <fstream>
using namespace std;


vector<intT> get_permutation(const vector<intT> &grouping)
{
    auto comp = [&](int i, int j)
    {
        return grouping[i] < grouping[j];
    };

    vector<intT> v(grouping.size());
    iota(v.begin(), v.end(), 0);
    sort (v.begin(), v.end(), comp);

    return v;
}

vector<intT> get_partition(const vector<intT> &grouping)
{
    vector<intT> reordered_groups(grouping);
    vector<intT> partition;

    sort(reordered_groups.begin(), reordered_groups.end());
    
    intT current_group = -1;
    intT current_size = 0;
    intT i = 0;
    while (i < reordered_groups.size())
    {
        if (current_group != reordered_groups[i])
        {
            current_group = reordered_groups[i];
            partition.push_back(i);
        }
        i++;
    }
    partition.push_back(reordered_groups.size());
    return partition;
}

vector<intT> get_fixed_size_grouping(const vector<intT> &grouping, intT row_block_size)
{
    vector<intT> result_grouping(grouping.size(), -1);
    vector<intT> perm = get_permutation(grouping);
    for (intT i = 0; i < perm.size(); i++)
    {
        result_grouping[perm[i]] = i/row_block_size;
    }
    return result_grouping;
}

bool check_structured_sparsity(vector<intT>& structured_sparsity_pattern, vector<intT>& structured_sparsity_column_counter, intT* row, intT row_len, int m)
{
    //check that Row does not break the m:n structured sparsity when added to structured_sparsity_pattern;
    intT i = 0;
    intT j = 0;
    while (i < structured_sparsity_pattern.size() && j < row_len)
    {
        if (structured_sparsity_pattern[i] < row[j])
            i++;
        else if (structured_sparsity_pattern[i] > row[j])
            j++;
        else
        {
            if (structured_sparsity_column_counter[i] >= m) //column is already full
                return false;
            i++;
            j++;
        }
    }

    return true;
}

void update_structured_sparsity(vector<intT>& structured_sparsity_pattern, vector<intT>& structured_sparsity_column_counter, intT* row, intT row_len)
{
    //A,B sparse rows (compressed indices format)
    vector<intT> new_pattern;
    vector<intT> new_counter;

    intT i = 0;
    intT j = 0;

    while (i < structured_sparsity_pattern.size() && j < row_len)
    {
        if (structured_sparsity_pattern[i] < row[j])
        {
            new_pattern.push_back(structured_sparsity_pattern[i]);
            new_counter.push_back(structured_sparsity_column_counter[i]);
            i++;
        }
        else if (structured_sparsity_pattern[i] > row[j])
        {
            new_pattern.push_back(row[j]);
            new_counter.push_back(1);
            j++;
        }
        else
        {
            new_pattern.push_back(structured_sparsity_pattern[i]);
            new_counter.push_back(structured_sparsity_column_counter[i] + 1);
            i++;
            j++;
        }
    }
    
    while (i < structured_sparsity_pattern.size())
    {
        new_pattern.push_back(structured_sparsity_pattern[i]);
        new_counter.push_back(structured_sparsity_column_counter[i]);
        i++;
    }
    
    while (j < row_len)
    {
        new_pattern.push_back(row[j]);
        new_counter.push_back(1);
        j++;
    }

    structured_sparsity_pattern.clear();
    structured_sparsity_column_counter.clear();
    copy(new_pattern.begin(), new_pattern.end(), std::back_inserter(structured_sparsity_pattern));
    copy(new_counter.begin(), new_counter.end(), std::back_inserter(structured_sparsity_column_counter));
}

vector<intT> OLD_merge_rows(vector<intT> A, intT*B, intT size_B)
{
    //A,B sparse rows (compressed indices format)
    vector<intT> result;
    intT size_A = A.size();

    result.reserve(A.size() + size_B);
    std::merge(A.begin(), A.end(),
            B, B + size_B,
           std::back_inserter(result));

    return result;
}

vector<intT> merge_rows(vector<intT> A, intT*B, intT size_B)
{

    vector<intT> result;
    auto i = A.begin();
    auto new_i = A.begin();
    intT j = 0;
    while(j < size_B)
    {
        auto B_val = B[j];

        //find j position in A
        new_i = std::lower_bound(i, A.end(), B_val);

        if (new_i == A.end()) break;


        //copy A up to there
        result.insert(result.end(), i, new_i);
        result.push_back(B_val);
        if (*new_i == B_val) new_i++;

        i = new_i;
        j++;
    }

    result.insert(result.end(), B + j, B + size_B);
    return result;
}

void save_blocking_data(ostream &outfile, CLineReader &cLine, BlockingEngine &bEngine, CSR &cmat, bool save_blocking, ostream &blocking_outfile)
{
    string header;
    string values;
    auto add_to_output = [&](string name, string value)
    {
        header += name + ","; 
        values += value + ",";
    };

    bEngine.CollectBlockingInfo(cmat);

    //matrix infos
    add_to_output("matrix", cLine.filename_);
    add_to_output("rows", to_string(cmat.rows));
    add_to_output("cols", to_string(cmat.cols));
    add_to_output("nonzeros", to_string(cmat.nztot()));
    add_to_output("symmetrize", to_string(cLine.symmetrize_));

    //blocking info
    add_to_output("blocking_algo", to_string(cLine.blocking_algo_));
    add_to_output("tau", to_string(cLine.tau_));
    add_to_output("row_block_size", to_string(cLine.row_block_size_));
    add_to_output("col_block_size", to_string(cLine.col_block_size_));
    add_to_output("use_pattern", to_string(cLine.sim_use_pattern_));
    add_to_output("sim_use_groups", to_string(cLine.sim_use_groups_));
    add_to_output("sim_measure", to_string(cLine.sim_measure_));
    add_to_output("reorder", to_string(cLine.reorder_));
    add_to_output("exp_name", cLine.exp_name_);

    //multiplication info
    add_to_output("b_cols", to_string(cLine.B_cols_));
    add_to_output("warmup", to_string(cLine.warmup_));
    add_to_output("exp_repetitions", to_string(cLine.exp_repetitions_));
    add_to_output("multiplication_algo", to_string(cLine.multiplication_algo_));
    add_to_output("n_streams", to_string(cLine.n_streams_));

    //blocking results
    add_to_output("time_to_block", to_string(bEngine.timer_total));
    add_to_output("time_to_merge", to_string(bEngine.timer_merges));
    add_to_output("time_to_compare", to_string(bEngine.timer_comparisons));

    add_to_output("VBR_nzcount", to_string(bEngine.VBR_nzcount));
    add_to_output("VBR_nzblocks_count", to_string(bEngine.VBR_nzblocks_count));
    add_to_output("VBR_average_height", to_string(bEngine.VBR_average_height));
    add_to_output("VBR_longest_row", to_string(bEngine.VBR_longest_row));


    add_to_output("merge_counter", to_string(bEngine.merge_counter));
    add_to_output("comparison_counter", to_string(bEngine.comparison_counter));
    add_to_output("average_merge_tau", to_string(bEngine.average_merge_tau));
    add_to_output("average_row_distance", to_string(bEngine.average_row_distance));

    //multiplication results
    add_to_output("avg_time_multiply", to_string(bEngine.multiplication_timer_avg));
    add_to_output("std_time_multiply", to_string(bEngine.multiplication_timer_std));

    outfile << header << endl;
    outfile << values << endl;

    if (save_blocking)
    {
        blocking_outfile << "GROUPING,";
        print_vec(bEngine.grouping_result, blocking_outfile, ",");
    }
}
