#include "OpMgr.hh"

/**
OpTest
================
**/

int main(int argc, char** argv)
{
    OpMgr op(argc, argv);
    op.snap();
    return 0 ; 
}

