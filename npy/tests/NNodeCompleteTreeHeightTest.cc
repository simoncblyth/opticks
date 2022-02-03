/**
NNodeCompleteTreeHeightTest.cc
===============================

Formerly NCSGDataTest 

See notes/issues/investigate-sams-x375-a-solid-that-breaks-balancing.rst

**/

#include "NNode.hpp"
#include "OPTICKS_LOG.hh"

void test_NumNodes()
{
    for(unsigned h=0 ; h < 20 ; h++) 
    {
        unsigned nn = nnode::NumNodes(h); 
        unsigned h2 = nnode::CompleteTreeHeight(nn); 
        std::cout 
            << " h " << std::setw(10) << h 
            << " NumNodes " << std::setw(20) << nn
            << " h2 " << std::setw(10) << h2 
            << std::endl 
            ; 
    }
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_NumNodes();

    return 0 ; 
}



