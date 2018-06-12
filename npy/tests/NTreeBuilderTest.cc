#include <cassert>
#include <vector>

#include "OPTICKS_LOG.hh"

#include "NBox.hpp"
#include "NTreeBuilder.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv); 

    std::vector<nnode*> prims(4) ; 

    nbox* a = new nbox(make_box3(400,400,400));
    nbox* b = new nbox(make_box3(500,100,100));
    nbox* c = new nbox(make_box3(100,500,100));
    nbox* d = new nbox(make_box3(100,100,500));

    prims[0] = a ; 
    prims[1] = b ; 
    prims[2] = c ; 
    prims[3] = d ; 

    nnode* root = NTreeBuilder::UnionTree(prims) ; 
    assert( root ) ; 
    root->dump();

    return 0 ; 
}


