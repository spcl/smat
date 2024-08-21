#include <fstream>
#include <iostream>
#include <algorithm>
#include "utilities.h"
#include "blocking.h"

using namespace std;

int main(int argc, char* argv[])
{
    CLineReader cli(argc, argv);
    intT block_size = cli.col_block_size_;

    vector<intT> row_A{1,2,5,10,12,20};
    intT row_B[] = {0,2,4,10,16};
    intT size_B = 5;

    float hamming_sim = HammingDistanceGroup(row_A,1,row_B,size_B,1,block_size);
    float jaccard_sim = JaccardDistanceGroup(row_A,1,row_B,size_B,1,block_size);

    float hamming_sim_openmp = HammingDistanceGroupOPENMP(row_A,1,row_B,size_B,1,block_size);
    float jaccard_sim_openmp = JaccardDistanceGroupOPENMP(row_A,1,row_B,size_B,1,block_size);

    cout << "block_size: " << block_size << endl;

    cout << "row A:  ";
    print_vec(row_A);

    cout << "row B:  ";
    print_mat(row_B, 1, size_B, 1);

    cout << " Hamming: " << hamming_sim << endl;
    cout << " Hamming(OPENMP): " << hamming_sim_openmp << endl;
 
    cout << " Jaccard: " << jaccard_sim << endl;
    cout << " Jaccard(OPENMP): " << jaccard_sim_openmp << endl;
}