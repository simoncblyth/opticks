/**
QPMTTest.cc
=================

QProp::lookup_scan is testing on GPU interpolation
BUT: the kernel is rather hidden away, need a more open test
to workout how to integrate with j/Layr/Layr.h TMM calcs

qprop.h is very simple, might as well extend that a little 
into a dedicated qpmt.h handling thickness, rindex, kindex

Dependency on "PMTSim/JPMT.h" which is actually ~/j/Layr/JPMT.h
-----------------------------------------------------------------

The ~/j/PMTSim/CMakeLists.txt "virtual" package installs 
~/j/Layr/JPMT.h into PMTSim install dirs that are used by 
this test within a PMTSim_FOUND block in its CMakeLists.txt

Rejig JPMT dependency
-----------------------

Former JPMT member removed from QPMTTest as generality is improved 
by having NP* rindex, thickness ctor args to QPMT 
avoiding the coupling with the NP_PROP_BASE  based 
JPMT that are seeking to replace. 

Moving from JPMT from text props to SSim/jpmt NPFold
-----------------------------------------------------

Whats missing from JPMT approach is contiguous pmt index array 
with category and qe_scale so can start from pmtid and get the pmtcat
and the qe for an energy.::

    jcv _PMTSimParamData
    ./Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/_PMTSimParamData.h


 
**/

#include "QPMT.hh"

template<typename T>
struct QPMTTest
{
    const QPMT<T>& qpmt ; 

    NP* domain ; 
    NP* rindex_interp ; 
    NP* qeshape_interp ; 
    NP* stackspec_interp ; 

    QPMTTest(const QPMT<T>& qpmt ); 

    void rindex_test(); 
    void qeshape_test(); 
    void stackspec_test(); 

    void save() const ; 
};

template<typename T>
QPMTTest<T>::QPMTTest(const QPMT<T>& qpmt_ )
    :
    qpmt(qpmt_),
    domain(NP::Linspace<T>( 1.55, 15.50, 1550-155+1 )), //  np.linspace( 1.55, 15.50, 1550-155+1 )  
    rindex_interp(nullptr),
    qeshape_interp(nullptr),
    stackspec_interp(nullptr)
{
}

template<typename T>
void QPMTTest<T>::rindex_test()
{
    rindex_interp = qpmt.rindex_interpolate(domain);   
}
template<typename T>
void QPMTTest<T>::qeshape_test()
{
    qeshape_interp = qpmt.qeshape_interpolate(domain);   
}
template<typename T>
void QPMTTest<T>::stackspec_test()
{
    stackspec_interp = qpmt.stackspec_interpolate(domain);   
}




template<typename T>
void QPMTTest<T>::save() const 
{
    qpmt.save("$FOLD") ; 
    domain->save("$FOLD/domain.npy"); 
    if(rindex_interp) rindex_interp->save("$FOLD/rindex_interp.npy" ); 
    if(qeshape_interp) qeshape_interp->save("$FOLD/qeshape_interp.npy" ); 
    if(stackspec_interp) stackspec_interp->save("$FOLD/stackspec_interp.npy" ); 
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
    OPTICKS_LOG(argc, argv); 

#ifdef WITH_JPMT
    std::cout << "QPMTTest.cc : WITH_JPMT " << std::endl ; 
    JPMT pmt ; 
    std::cout << pmt.desc() << std::endl ;
    pmt.save("$FOLD") ; 
    const NP* rindex = pmt.rindex ; 
    const NP* thickness = pmt.thickness ;
    const NP* qeshape = pmt.qeshape ;  
    const NP* lcqs = pmt.lcqs ;        // nullptr
#else
    std::cout << "QPMTTest.cc : NOT-WITH_JPMT " << std::endl ; 
    SPMT* pmt = SPMT::Load();
    if(pmt == nullptr)
    {
        std::cout << "QPMTTest.cc FAILED TO SPMT::Load ? IS GEOM envvar defined ? " << std::endl ; 
        return 1 ; 
    }

    const NP* rindex = pmt->rindex ; 
    const NP* thickness = pmt->thickness ;
    const NP* qeshape = pmt->qeshape ;  
    const NP* lcqs = pmt->lcqs ; 
#endif

    QPMT<float> qpmt(rindex, thickness, qeshape, lcqs ) ; 
    std::cout << qpmt.desc() << std::endl ; 

    QPMTTest<float> t(qpmt); 

    t.rindex_test(); 
    t.qeshape_test(); 
    t.stackspec_test(); 
 
    cudaDeviceSynchronize();
    t.save();  

    return 0 ; 
}
