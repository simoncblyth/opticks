/**
CSGOptiXTMTest
===============

**/

#include "OPTICKS_LOG.hh"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    return CSGOptiX::SimtraceMain();
}



