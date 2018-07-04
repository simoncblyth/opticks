#include "OKMgr.hh"
#include "NGPU.hpp"
#include "OPTICKS_LOG.hh"

/**
OKTest
================
**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    OKMgr ok(argc, argv);
    ok.propagate();
    ok.visualize();

    //NGPU::GetInstance()->saveBuffer("$TMP/OKTest_NGPU.npy");  

    return ok.rc();
}

