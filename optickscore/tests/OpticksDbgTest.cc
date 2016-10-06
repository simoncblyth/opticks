#include <cassert>

#include "Opticks.hh"

#include "PLOG.hh"
#include "OKCORE_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKCORE_LOG__ ; 

    Opticks ok(argc, argv, "--dindex 1,10,100,-200");
    ok.configure();

    assert(ok.isDbgPhoton(1) == true );
    assert(ok.isDbgPhoton(10) == true );
    assert(ok.isDbgPhoton(100) == true );
    assert(ok.isDbgPhoton(-200) == true );


    const std::vector<int>& dindex = ok.getDbgIndex();

    assert(dindex.size() == 4);
    assert(dindex[0] == 1);
    assert(dindex[1] == 10);
    assert(dindex[2] == 100);
    assert(dindex[3] == -200);



    return 0 ; 
}
