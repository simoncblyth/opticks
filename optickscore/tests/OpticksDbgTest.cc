
// OpticksDbgTest --OKCORE trace

#include <cassert>

#include "NPY.hpp"
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

void test_getMaskBuffer(int argc, char** argv)
{
    Opticks ok(argc, argv, "--maskindex 1,3,5,7,9");
    ok.configure();

    NPY<unsigned>* msk = ok.getMaskBuffer() ;

    assert( msk && msk->getShape(0) == 5 );
    msk->dump("msk");
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKCORE_LOG__ ; 

    //test_isDbgPhoton_string(argc, argv);
    //test_isDbgPhoton_path(argc, argv);
    test_getMaskBuffer(argc, argv);

    return 0 ; 
}
