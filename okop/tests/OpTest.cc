#include "Opticks.hh"
#include "OpMgr.hh"

/**
OpTest
================
**/

int main(int argc, char** argv)
{
    Opticks ok(argc, argv, "--tracer"); 
    OpMgr op(&ok);
    op.snap();
    return 0 ; 
}

