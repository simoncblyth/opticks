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
#include "NPFold.h"
#include "QPMT.hh"

template<typename T>
struct QPMTTest
{
    static constexpr const char* LPMTID_LIST = "0,10,55,98,100,137,1000,10000,17611" ; 

    const QPMT<T>& qpmt ; 

    const char* lpmtid_list ; 
    NP* lpmtid ; 
    NP* energy_eV_domain ; 
    int num_mct ; 
    NP* mct_domain ; 


    QPMTTest(const QPMT<T>& qpmt ); 

    NPFold* make_qscan(); 

};

template<typename T>
QPMTTest<T>::QPMTTest(const QPMT<T>& qpmt_ )
    :
    qpmt(qpmt_),
    lpmtid_list(ssys::getenvvar("LPMTID_LIST", LPMTID_LIST)), // pick some lpmtid (<17612) 
    lpmtid(NPX::FromString<int>(lpmtid_list,',')), 
    energy_eV_domain(NP::Linspace<T>( 1.55, 15.50, 1550-155+1 )), //  np.linspace( 1.55, 15.50, 1550-155+1 )  
    num_mct(ssys::getenvint("NUM_MCT",181)),
    mct_domain(NP::MakeWithType<T>(NP::MinusCosThetaLinearAngle<double>(num_mct)))
{
}


template<typename T>
NPFold* QPMTTest<T>::make_qscan()
{
    NPFold* qscan = new NPFold ; 

    qscan->add("energy_eV_domain", energy_eV_domain ) ; 
    qscan->add("mct_domain", mct_domain ) ; 
    qscan->add("lpmtid", lpmtid ) ; 

    qscan->add("lpmtcat_rindex", qpmt.lpmtcat_rindex(energy_eV_domain) ) ; 
    qscan->add("lpmtcat_qeshape", qpmt.lpmtcat_qeshape(energy_eV_domain) ) ; 
    //qscan->add("lpmtcat_stackspec", qpmt.lpmtcat_stackspec(energy_eV_domain) ) ; 
    //qscan->add("lpmtid_stackspec", qpmt.lpmtid_stackspec(energy_eV_domain) ) ; 

    qscan->add("lpmtid_ART", qpmt.lpmtid_ART(mct_domain, lpmtid) ) ; 
    qscan->add("lpmtid_ARTE", qpmt.lpmtid_ARTE(mct_domain, lpmtid) ) ; 

    return qscan ; 
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

    QPMT<float> qp(rindex, thickness, qeshape, lcqs ) ; 
    std::cout << qp.desc() << std::endl ; 

    NPFold* qpf = qp.get_fold(); 
    qpf->save("$FOLD/qpmt"); 

    QPMTTest<float> qpt(qp); 
    NPFold* qscan = qpt.make_qscan(); 
 
    cudaDeviceSynchronize();
    qscan->save("$FOLD/qscan");  

    return 0 ; 
}
