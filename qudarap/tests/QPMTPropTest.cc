/**
QPMTPropTest.cc
=================

QProp::lookup_scan is testing on GPU interpolation
BUT: the kernel is rather hidden away, need a more open test
to workout how to integrate with j/Layr/Layr.h TMM calcs

qprop.h is very simple, might as well extend that a little 
into a qpmtprop.h

**/

#include <cuda_runtime.h>
#include "JPMTProp.h"
#include "QPMTProp.hh"
#include "OPTICKS_LOG.hh"

const char* FOLD = "/tmp/QPMTPropTest" ;

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    JPMTProp pp ; 
    //std::cout << pp.desc() << std::endl ;

    QPMTProp<double> qpp(pp.rindex, pp.thickness) ;   
    std::cout << qpp.desc() << std::endl ; 

    qpp.rindex->lookup_scan( 1.55, 15.5, 100u, FOLD, "rindex" );   
    
    cudaDeviceSynchronize();

    return 0 ; 
}
