#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "OpMgr.hh"

/**
OpTest
================
**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv, "--tracer");
    OpMgr op(&ok);
    op.snap();
    return 0 ; 
}

