#include "OKMgr.hh"
#include "OPTICKS_LOG.hh"

/**
OKTest
================
**/

int main(int argc, char** argv)
{
    const char* pfx = "OKTest" ; 

    for(unsigned i=0 ; i < argc ; i++) std::cout << pfx << ".a " <<  argv[i] << std::endl ; 

    OPTICKS_LOG(argc, argv); 

    for(unsigned i=0 ; i < argc ; i++) std::cout << pfx << ".b " <<  argv[i] << std::endl ; 

    OKMgr ok(argc, argv);

    for(unsigned i=0 ; i < argc ; i++) std::cout << pfx << ".c " <<  argv[i] << std::endl ; 

    ok.propagate();

    ok.visualize();

    return ok.rc();
}

