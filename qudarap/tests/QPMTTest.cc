/**
QPMTTest.cc
=================

QProp::lookup_scan is testing on GPU interpolation
BUT: the kernel is rather hidden away, need a more open test
to workout how to integrate with j/Layr/Layr.h TMM calcs


qprop.h is very simple, might as well extend that a little 
into a dedicated qpmt.h handling thickness, rindex, kindex

**/

#include <cuda_runtime.h>
#include "JPMT.h"
#include "QPMT.hh"
#include "OPTICKS_LOG.hh"

const char* FOLD = "/tmp/QPMTTest" ;

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    JPMT jp ; 
    //std::cout << jp.desc() << std::endl ;

    QPMT<double> qp(jp.rindex, jp.thickness) ;   
    std::cout << qp.desc() << std::endl ; 

    qp.rindex_prop->lookup_scan( 1.55, 15.5, 100u, FOLD, "rindex" );   
    
    cudaDeviceSynchronize();

    return 0 ; 
}
