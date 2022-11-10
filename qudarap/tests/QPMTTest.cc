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

    JPMT pmt ; 
    //std::cout << pmt.desc() << std::endl ;

    QPMT<float> qp(pmt.rindex, pmt.thickness) ;   
    std::cout << qp.desc() << std::endl ; 
    qp.save(FOLD) ; 

    //std::cout << "qp.rindex_prop->lookup_scan" << std::endl ;  
    //qp.rindex_prop->lookup_scan( 1.55, 15.5, 100u, FOLD, "rindex" );   

    std::cout << "qp.interpolate" << std::endl ;  

    NP* domain = NP::Linspace<float>( 1.55, 15.5, 4 );  
    NP* interp = qp.interpolate(domain ); 
    interp->save(FOLD, "interp.npy" ); 

    std::cout << " interp.sstr " << interp->sstr() << std::endl ; 
    
    cudaDeviceSynchronize();

    return 0 ; 
}
