/**
QPMT_Test.cc 
=================
**/

#include "QPMT.hh"

template<typename T>
struct QPMT_Test
{
    const QPMT<T>& qpmt ; 

    NP* domain ; 
    NP* rindex_interp ; 
    NP* qeshape_interp ; 

    QPMT_Test(const QPMT<T>& qpmt ); 

    void rindex_test(); 
    void qeshape_test(); 
    void save() const ; 
};

template<typename T>
QPMT_Test<T>::QPMT_Test(const QPMT<T>& qpmt_ )
    :
    qpmt(qpmt_),
    domain(NP::Linspace<T>( 1.55, 15.50, 1550-155+1 )), //  np.linspace( 1.55, 15.50, 1550-155+1 )  
    rindex_interp(nullptr),
    qeshape_interp(nullptr)
{
}

template<typename T>
void QPMT_Test<T>::rindex_test()
{
    rindex_interp = qpmt.rindex_interpolate(domain);   
}
template<typename T>
void QPMT_Test<T>::qeshape_test()
{
    qeshape_interp = qpmt.qeshape_interpolate(domain);   
}


template<typename T>
void QPMT_Test<T>::save() const 
{
    qpmt.save("$FOLD") ; 
    domain->save("$FOLD/domain.npy"); 
    if(rindex_interp) rindex_interp->save("$FOLD/rindex_interp.npy" ); 
    if(qeshape_interp) qeshape_interp->save("$FOLD/qeshape_interp.npy" ); 
}


#include <cuda_runtime.h>

#ifdef WITH_JPMT
#include "JPMT.h"
#else
#include "SPMT.h"
#endif


#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    //assert(0 && "ABANDONED FOR NOW : SEE NOTE IN QPMT_Test.sh " ); 

    OPTICKS_LOG(argc, argv); 

#ifdef WITH_JPMT
    std::cout << "QPMT_Test.cc : WITH_JPMT " << std::endl ; 
    JPMT pmt ; 
    std::cout << pmt.desc() << std::endl ;
    pmt.save("$FOLD") ; 
    const NP* rindex = pmt.rindex ; 
    const NP* thickness = pmt.thickness ;
    const NP* qeshape = pmt.qeshape ;  
    const NP* lcqs = pmt.lcqs ;        // nullptr
#else
    std::cout << "QPMT_Test.cc : NOT-WITH_JPMT " << std::endl ; 
    SPMT* pmt = SPMT::Load();
    if(pmt == nullptr)
    {
        std::cout << "QPMT_Test.cc FAILED TO SPMT::Load ? IS GEOM envvar defined ? " << std::endl ; 
        return 1 ; 
    }

    const NP* rindex = pmt->rindex ; 
    const NP* thickness = pmt->thickness ;
    const NP* qeshape = pmt->qeshape ;  
    const NP* lcqs = pmt->lcqs ; 
#endif

    QPMT<float> qpmt(rindex, thickness, qeshape, lcqs ) ; 
    std::cout << qpmt.desc() << std::endl ; 

    QPMT_Test<float> t(qpmt); 
    t.rindex_test(); 
    t.qeshape_test(); 
 
    cudaDeviceSynchronize();
    t.save();  

    return 0 ; 
}
