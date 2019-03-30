// TEST=NNodeDumpTest om-t 

#include "NNode.hpp"
#include "NNodeSample.hpp"

#include "OPTICKS_LOG.hh"

typedef std::vector<nnode*> VN ;


void test_dump(const VN& nodes, unsigned idx)
{
    assert( idx < nodes.size() ) ; 
    nnode* n = nodes[idx] ; 
    LOG(info) << "\n" << " sample idx : " << idx ; 
    n->dump(); 
}

void test_dump(const VN& nodes)
{
    for(unsigned i=0 ; i < nodes.size() ; i++) test_dump(nodes, i) ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);

    int idx = argc > 1 ? atoi(argv[1]) : -1 ; 

    if( idx == -1 ) 
    {
        test_dump(nodes); 
    }
    else
    {
        test_dump(nodes, idx); 
    }

    return 0 ; 
}



