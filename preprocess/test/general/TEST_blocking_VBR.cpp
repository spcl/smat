#include "matrices.h"
#include "blocking.h"
#include "utilities.h"

#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
    CLineReader cli(argc, argv);
    if (cli.verbose_ > 0) cli.print();
    CSR cmat(cli);
    if (cli.verbose_ > 0) cmat.print(cli.verbose_);
    BlockingEngine bEngine(cli);

    //evaluate the grouping
    if (cli.verbose_ > 0) cout << "Blocking and reordering the matrix." <<endl;
    bEngine.GetGrouping(cmat);

    if (cli.verbose_ > 1) cout << "Grouping of the rows: "; 
    if (cli.verbose_ > 1) print_vec(bEngine.grouping_result);
    
    bEngine.print();

    //CREATE VBR FROM GROUPING
    if (cli.verbose_ > 0) cout << "Filling a VBR with nonzero blocks." << endl;
    //create a VBR matrix from grouping (without reordering the original csr)
    VBR vbmat;
    vbmat.fill_from_CSR_inplace(cmat, bEngine.grouping_result, cli.col_block_size_, cli.row_block_size_, cli.force_fixed_size);
    if (cli.verbose_ > 0) vbmat.print(cli.verbose_);

    //GET BLOCK PROPERTIES FROM GROUPING WITHOUT CREATING VBR EXPLICITLY
    bEngine.CollectBlockingInfo(cmat);
    
    cout << "BLOCKING RESULTS:" << endl;
    save_blocking_data(cout, cli, bEngine, cmat, false);

    cout << "TEST COMPLETED" << endl;
}