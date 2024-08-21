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
    BlockingEngine bEngine(cli);

    
    //evaluate the grouping
    vector<intT> grouping = bEngine.GetGrouping(cmat); 
    if (cli.verbose_ > 0) bEngine.print();

    ofstream outfile;
    bool save_grouping = true;
    outfile.open(cli.outfile_);

    ofstream outfile_grouping;
    if (save_grouping)
    {
        outfile_grouping.open(cli.outfile_ + ".g");
    }
    save_blocking_data(outfile, cli, bEngine, cmat, save_grouping, outfile_grouping);

    if (cli.verbose_ > 0)
    {
        save_blocking_data(cout, cli, bEngine, cmat, false, outfile_grouping);
    }

    bool save_reordered_matrix = true;
    if (save_reordered_matrix) {
        cmat.reorder(grouping);
        outfile.close();
        size_t lastindex = cli.filename_.find_last_of(".");
        string rawname = cli.filename_.substr(0, lastindex);
        outfile.open(rawname + "_reordered.el");
        cmat.save_to_edgelist(outfile, " ", true, 0);
    }
    
    if (cli.verbose_> 0) cout << "Nonzero Blocks:" << bEngine.VBR_nzblocks_count << endl;
}
