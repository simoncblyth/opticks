/**

See notes/issues/investigate-sams-x375-a-solid-that-breaks-balancing.rst

**/

#include "NCSGData.hpp"
#include "OPTICKS_LOG.hh"

void test_NumNodes()
{
    for(unsigned h=0 ; h < 260 ; h++) 
    {
        unsigned nn = NCSGData::NumNodes(h); 
        std::cout 
            << " h " << std::setw(10) << h 
            << " NumNodes " << std::setw(20) << nn
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



