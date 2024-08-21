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
    if (cli.verbose_ > 1) cmat.print(cli.verbose_);
    
    vector<float> taus = {0.9f, 0.8f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.01f};
    //evaluate the grouping


    float best_tau = -1;
    float best_blocks = cmat.nztot();
    for (auto tau : taus)
    {
        BlockingEngine bEngine(cli);
        bEngine.tau = tau;
        bEngine.GetGrouping(cmat); 
        bEngine.CollectBlockingInfo(cmat);
        if (bEngine.VBR_nzblocks_count <= best_blocks)
        {
            best_tau = tau;
            best_blocks = bEngine.VBR_nzblocks_count;
        }
    }

    BlockingEngine bEngine(cli);
    bEngine.tau = best_tau;
    cout << "best tau found: " << best_tau << " with " << best_blocks << " nz blocks." << endl;
    bEngine.GetGrouping(cmat);
    cmat.reorder(bEngine.grouping_result);
    cout << "REORDERED MATRIX: " << endl;
    if (cli.verbose_ > 1) cmat.print(cli.verbose_);


    ofstream graphFile(cli.outfile_ + ".g");
    cmat.save_to_edgelist(graphFile);

    ofstream outfile;
    outfile.open(cli.outfile_);
    ofstream outfile_grouping;
    save_blocking_data(outfile, cli, bEngine, cmat, false, outfile_grouping);

    if (cli.verbose_ > 0)
    {
        save_blocking_data(cout, cli, bEngine, cmat, false, outfile_grouping);
    }
}