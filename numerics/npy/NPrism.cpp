#include "NPrism.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"

#include <cmath>
#include <cassert>


void nprism::dump(const char* msg)
{
    param.dump(msg);
}



npart nprism::part()
{
    // hmm more dupe of hemi-pmt.cu/make_prism
    // but if could somehow make vector types appear 
    // the same could use same code with CUDA ?

    float h  = height();
    float hw = hwidth();
    float d  = depth();

    nbbox bb ;
    bb.min = {-hw,0.f,-d/2.f } ;
    bb.max = { hw,  h, d/2.f } ;

    npart p ; 
    p.zero();            

    p.setParam(param) ; 
    p.setTypeCode(PRISM); 
    p.setBBox(bb);

    return p ; 
}


