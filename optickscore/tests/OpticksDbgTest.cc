
// OpticksDbgTest --OKCORE trace

#include <cassert>

#include "Opticks.hh"

#include "PLOG.hh"
#include "OKCORE_LOG.hh"


void test_isDbgPhoton_string(int argc, char** argv)
{
    Opticks ok(argc, argv, "--dindex 1,10,100,200");
    ok.configure();

    assert(ok.isDbgPhoton(1) == true );
    assert(ok.isDbgPhoton(10) == true );
    assert(ok.isDbgPhoton(100) == true );
    assert(ok.isDbgPhoton(200) == true );

    const std::vector<unsigned>& dindex = ok.getDbgIndex();

    assert(dindex.size() == 4);
    assert(dindex[0] == 1);
    assert(dindex[1] == 10);
    assert(dindex[2] == 100);
    assert(dindex[3] == 200);
}



void test_isDbgPhoton_path(int argc, char** argv)
{
    Opticks ok(argc, argv, "--dindex $TMP/c.npy");
    ok.configure();
    if(ok.getNumDbgPhoton() == 0 ) return ; 



    assert(ok.isDbgPhoton(268) == true );
    assert(ok.isDbgPhoton(267) == false );
}





int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKCORE_LOG__ ; 

    //test_isDbgPhoton_string(argc, argv);

    test_isDbgPhoton_path(argc, argv);



    return 0 ; 
}
