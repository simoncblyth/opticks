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

Moving from JPMT from text props to SSim/jpmt NPFold based SPMT.h 
----------------------------------------------------------------------

Whats missing from JPMT approach is contiguous pmt index array 
with category and qe_scale so can start from pmtid and get the pmtcat
and the qe for an energy.::

    jcv _PMTSimParamData
    ./Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/_PMTSimParamData.h



Find some lpmtid with lpmtcat == 0 
-------------------------------------

::

    In [30]: np.where( t.src_lcqs[:,0] == 0 )[0]
    Out[30]: array([   55,    98,   137,   267,   368, ..., 17255, 17327, 17504, 17526, 17537])

    In [31]: np.where( t.src_lcqs[:,0] == 0 )[0].shape
    Out[31]: (2720,)
 
**/

#include "ssys.h"
#include "NPX.h"
#include "QPMT.hh"

template<typename T>
struct QPMTTest
{
    static constexpr const char* LPMTID_LIST = "0,10,55,98,100,137,1000,10000,17611" ; 

    const QPMT<T>& qpmt ; 

    const char* lpmtid_list ; 
    NP* lpmtid ; 
    NP* energy_eV_domain ; 
    NP* mct_domain ; 

    NP* lpmtcat_rindex ; 
    NP* lpmtcat_qeshape ; 
    NP* lpmtcat_stackspec ; 

    NP* lpmtid_stackspec ; 
    NP* lpmtid_ART ; 
    NP* lpmtid_ARTE ; 

    // HMM: switch to NPFold ? 


    QPMTTest(const QPMT<T>& qpmt ); 

    void lpmtcat_rindex_test(); 
    void lpmtcat_qeshape_test(); 
    void lpmtcat_stackspec_test(); 

    void lpmtid_stackspec_test(); 
    void lpmtid_ART_test(); 
    void lpmtid_ARTE_test(); 

    void save(const char* base) const ; 
};

template<typename T>
QPMTTest<T>::QPMTTest(const QPMT<T>& qpmt_ )
    :
    qpmt(qpmt_),
    lpmtid_list(ssys::getenvvar("LPMTID_LIST", LPMTID_LIST)), // pick some lpmtid (<17612) 
    lpmtid(NPX::FromString<int>(lpmtid_list,',')), 
    energy_eV_domain(NP::Linspace<T>( 1.55, 15.50, 1550-155+1 )), //  np.linspace( 1.55, 15.50, 1550-155+1 )  
    mct_domain(NP::Linspace<T>(-1.0, 1.0, 180+1 )),
    lpmtcat_rindex(nullptr),
    lpmtcat_qeshape(nullptr),
    lpmtcat_stackspec(nullptr),
    lpmtid_stackspec(nullptr),
    lpmtid_ART(nullptr),
    lpmtid_ARTE(nullptr)

{
}

template<typename T>
void QPMTTest<T>::lpmtcat_rindex_test()
{
    lpmtcat_rindex = qpmt.lpmtcat_rindex(energy_eV_domain);   
}
template<typename T>
void QPMTTest<T>::lpmtcat_qeshape_test()
{
    lpmtcat_qeshape = qpmt.lpmtcat_qeshape(energy_eV_domain);   
}
template<typename T>
void QPMTTest<T>::lpmtcat_stackspec_test()
{
    lpmtcat_stackspec = qpmt.lpmtcat_stackspec(energy_eV_domain);   
}



template<typename T>
void QPMTTest<T>::lpmtid_stackspec_test()
{
    lpmtid_stackspec = qpmt.lpmtid_stackspec(energy_eV_domain, lpmtid);   
}

template<typename T>
void QPMTTest<T>::lpmtid_ART_test()
{
    lpmtid_ART = qpmt.lpmtid_ART(mct_domain, lpmtid);   
}
template<typename T>
void QPMTTest<T>::lpmtid_ARTE_test()
{
    lpmtid_ARTE = qpmt.lpmtid_ARTE(mct_domain, lpmtid);   
}



template<typename T>
void QPMTTest<T>::save(const char* base) const 
{
    qpmt.save(base) ; 
    energy_eV_domain->save(base, "energy_eV_domain.npy"); 
    mct_domain->save(base, "mct_domain.npy"); 
    lpmtid->save(base, "lpmtid.npy"); 
    if(lpmtcat_rindex) lpmtcat_rindex->save(base, "lpmtcat_rindex.npy" ); 
    if(lpmtcat_qeshape) lpmtcat_qeshape->save(base, "lpmtcat_qeshape.npy" ); 
    if(lpmtcat_stackspec) lpmtcat_stackspec->save(base, "lpmtcat_stackspec.npy" ); 
    if(lpmtid_stackspec) lpmtid_stackspec->save(base, "lpmtid_stackspec.npy" ); 
    if(lpmtid_ART) lpmtid_ART->save(base, "lpmtid_ART.npy" ); 
    if(lpmtid_ARTE) lpmtid_ARTE->save(base, "lpmtid_ARTE.npy" ); 
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

    /*
    t.lpmtcat_rindex_test(); 
    t.lpmtcat_qeshape_test(); 
    t.lpmtcat_stackspec_test(); 

    t.lpmtid_stackspec_test(); 
    t.lpmtid_ART_test(); 
    t.lpmtid_ARTE_test(); 
    */

    t.lpmtid_ART_test(); 
 
    cudaDeviceSynchronize();
    t.save("$FOLD");  

    return 0 ; 
}
