// okop/tests/OpTest.cc
#include "OpMgr.hh"

/**
OpTest
================
**/

int main(int argc, char** argv)
{
    OpMgr op(argc, argv, "--tracer");
    op.snap();
    return 0 ; 
}

