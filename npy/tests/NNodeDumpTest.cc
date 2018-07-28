
#include "NNode.hpp"
#include "NNodeSample.hpp"

#include "OPTICKS_LOG.hh"


void test_dump()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {   
        nnode* n = *it ; 
        n->dump();
    }
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_dump();

    return 0 ; 
}



