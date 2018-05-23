
#include "NNode.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


void test_dump()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    nnode::Tests(nodes);

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {   
        nnode* n = *it ; 
        n->dump();
    }
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    test_dump();

    return 0 ; 
}



