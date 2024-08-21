#include "matrices.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "utilities.h"

using namespace std;

int main(int argc, char* argv[])
{
    CLineReader cli(argc, argv);
    if (cli.verbose_ > 0) cli.print();
    CSR cmat(cli);
    if (cli.verbose_ > 0) cmat.print(cli.verbose_);
    if (cli.blocking_algo_ != fixed_size)
    {
        cout << "WARNING: no blocking allowed. changing algorithm to fixed_size, no reordering" << endl;
        cli.blocking_algo_ = fixed_size;
    }

    BlockingEngine bEngine(cli);
    bEngine.GetGrouping(cmat);
    VBR vbmat;
    
    vbmat.fill_from_CSR_inplace(cmat, bEngine.grouping_result, cli.col_block_size_);
    if (cli.verbose_ > 1) vbmat.print();



    intT B_rows = cmat.cols;
    intT B_cols = 5;
    DataT* mat_B = new DataT[B_cols*B_rows];
    fill_n(mat_B,B_cols*B_rows,1);

    cout << "B:" << endl;
    cout << B_cols << endl;
    cout << B_rows << endl;
    print_mat(mat_B,B_rows,B_cols, B_rows);


    intT C_rows = cmat.rows;
    intT C_cols = B_cols;
    DataT* mat_C = new DataT[C_rows*C_cols]{0};
    cmat.multiply(mat_B,B_cols,mat_C);

    print_mat(mat_C,cmat.rows,B_cols, cmat.rows);

    DataT* mat_C_VBR = new DataT[C_rows*C_cols]{0};
    vbmat.multiply(mat_B, B_cols, mat_C_VBR);

    print_mat(mat_C_VBR,cmat.rows,B_cols, cmat.rows);
    
    bool equality_check = equal(mat_C_VBR, mat_C_VBR + C_cols*C_rows, mat_C);
    cout << "CORRECTNESS CHECK: " << equality_check << endl;

    cout << "TEST COMPLETED" << endl;

}