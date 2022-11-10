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

struct QPMTTest
{
    static const char* FOLD ; 
    const JPMT& jpmt ; 
    QPMT<float> qpmt ; 

    NP* domain ; 
    NP* interp ; 

    QPMTTest(const JPMT& jpmt ); 
    void save() const ; 
};

const char* QPMTTest::FOLD = U::GetEnv("FOLD", "/tmp/QPMTTest") ;

QPMTTest::QPMTTest(const JPMT& jpmt_ )
    :
    jpmt(jpmt_),
    qpmt(jpmt.rindex, jpmt.thickness),
    domain(NP::Linspace<float>( 1.55, 15.5, 1550-155+1 )),
    interp(qpmt.interpolate(domain))
{
    std::cout << pmt.desc() << std::endl ;
    std::cout << qpmt.desc() << std::endl ; 
}

void QPMTTest::save() const 
{
    qpmt.save(FOLD) ; 
    qpmt.save(FOLD) ; 
    interp->save(FOLD, "interp.npy" ); 
    domain->save(FOLD, "domain.npy" ); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    JPMT jpmt ; 

    QPMTTest t(jpmt); 
    cudaDeviceSynchronize();
    t.save();  

    return 0 ; 
}
