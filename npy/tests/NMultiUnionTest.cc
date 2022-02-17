#include "NSphere.hpp"
#include "NNode.hpp"
#include "NMultiUnion.hpp"
#include "NBBox.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    nsphere* a = nsphere::Create(0.f,0.f,-50.f,100.f);
    nsphere* b = nsphere::Create(0.f,0.f, 50.f,100.f);
    nunion* u = nunion::make_union( a, b );

    nmultiunion* n = nmultiunion::CreateFromTree(CSG_CONTIGUOUS, u ); 
    assert( n ); 

    n->pdump(); 

    nbbox bb = n->bbox(); 
    LOG(info) << " bb.desc " << bb.desc() ;  

    return 0 ; 
}



