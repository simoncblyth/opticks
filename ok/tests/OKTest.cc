#include "OKMgr.hh"
#include "OPTICKS_LOG.hh"

/**
OKTest
================
**/

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv); 
 
    OKMgr ok(argc, argv);
    ok.propagate();
    ok.visualize();

    return ok.rc();
}

