#pragma once
#include <vector>
#include <string>
#include "matrices.h"
#include "input.h"
#include <fstream>
#include "definitions.h"

class BlockingEngine
{
    public:
        float tau = 0.5;

        intT col_block_size = 1;
        intT row_block_size = 1; //only used for fixed size blocking        
        bool use_groups = false;
        bool use_pattern = true;
        bool force_fixed_size = false;
        
        int structured_m = 2;
        int structured_n = 4;

        BlockingType blocking_algo = iterative_clocked;

        //bool blocking_completed = false;

        //measuring variables
        intT comparison_counter = 0;
        intT merge_counter = 0;
        float timer_total = 0;
        float timer_comparisons = 0;
        float timer_merges = 0;
        float average_row_distance = 0;
        float average_merge_tau = 0; 

        float multiplication_timer_avg = 0;
        float multiplication_timer_std = 0;

        intT VBR_nzcount = 0;
        intT VBR_nzblocks_count = 0;
        float VBR_average_height = 0;
        intT VBR_longest_row = 0;

        distFuncGroup comparator;
        std::vector<intT> grouping_result;

        std::vector<intT> GetGrouping(const CSR& cmat);
        void CollectBlockingInfo(const CSR& cmat);
        
        void SetComparator(int choice);
        
        void print();

        BlockingEngine(){};
        BlockingEngine(CLineReader &cline);
};

std::vector<intT> IterativeBlockingPatternCLOCKED(const CSR& cmat, float tau, distFuncGroup distanceFunction,intT block_size, bool use_size, bool use_pattern, intT &comparison_counter, intT &merge_counter, float &average_row_distance, float& average_merge_tau, float &timer_total, float &timer_comparisons, float &timer_merges);
std::vector<intT> IterativeBlockingPattern(const CSR& cmat, float tau, distFuncGroup distanceFunction,intT block_size, bool use_size, bool use_pattern,  intT &comparison_counter, intT &merge_counter, float &timer_total);
std::vector<intT> IterativeBlockingPatternMN(const CSR& cmat, float tau, distFuncGroup distanceFunction,intT block_size, bool use_size, bool use_pattern, int structured_m, int structured_n, intT &comparison_counter, intT &merge_counter, float &timer_total);
std::vector<intT> FixedBlocking(const CSR& cmat, intT row_block_size);
std::vector<intT> IterativeBlockingMaxSize(const CSR& cmat, float tau, distFuncGroup distanceFunction,intT col_block_size, intT max_row_block_size, bool use_size, bool use_pattern, intT &comparison_counter, intT &merge_counter, float &average_row_distance, float& average_merge_tau, float &timer_total, float &timer_comparisons, float &timer_merges);
std::vector<intT> IterativeBlockingQueue(const CSR& cmat, float tau, distFuncGroup distanceFunction,intT block_size, bool use_size, bool use_pattern, intT &comparison_counter, intT &merge_counter, float &average_row_distance, float& average_merge_tau, float &timer_total, float &timer_comparisons, float &timer_merges);



float HammingDistanceGroupOPENMP(std::vector<intT> row_A, intT group_size_A, intT* row_B, intT size_B, intT group_size_B, intT block_size);
float HammingDistanceGroup(std::vector<intT> row_A, intT group_size_A, intT* row_B, intT size_B, intT group_size_B, intT block_size);

float JaccardDistanceGroupOPENMP(std::vector<intT> row_A, intT group_size_A, intT* row_B, intT size_B, intT group_size_B, intT block_size);
float JaccardDistanceGroup(std::vector<intT> row_A, intT group_size_A, intT* row_B, intT size_B, intT group_size_B, intT block_size);

