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
    NP* interp ; 

    QPMTTest(const QPMT<T>& qpmt ); 
    void save() const ; 
};

template<typename T>
QPMTTest<T>::QPMTTest(const QPMT<T>& qpmt_ )
    :
    qpmt(qpmt_),
    domain(NP::Linspace<T>( 1.55, 15.5, 1550-155+1 )),
    interp(qpmt.interpolate(domain))
{
}

template<typename T>
void QPMTTest<T>::save() const 
{
    qpmt.save("$FOLD") ; 
    interp->save("$FOLD/interp.npy" ); 
    domain->save("$FOLD/domain.npy" ); 
}


#include <cuda_runtime.h>
#include "JPMT.h"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    JPMT jpmt ; 
    std::cout << jpmt.desc() << std::endl ;
    jpmt.save("$FOLD") ; 
    const NP* rindex = jpmt.rindex ; 
    const NP* thickness = jpmt.thickness ;


    QPMT<float> qpmt(rindex, thickness) ; 
    std::cout << qpmt.desc() << std::endl ; 

    QPMTTest<float> t(qpmt); 
    cudaDeviceSynchronize();
    t.save();  

    return 0 ; 
}
