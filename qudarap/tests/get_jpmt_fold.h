#pragma once
/**
get_jpmt_fold.h : NB SPMT is the way to go, JPMT is just for backwards check
==============================================================================

Dependency on "PMTSim/JPMT.h" which is actually ~/j/Layr/JPMT.h
-----------------------------------------------------------------

The ~/j/PMTSim/CMakeLists.txt "virtual" package installs 
~/j/Layr/JPMT.h into PMTSim install dirs that are used by 
this test within a PMTSim_FOUND block in its CMakeLists.txt

Moving from JPMT from text props to SSim/jpmt NPFold based SPMT.h 
----------------------------------------------------------------------

Whats missing from JPMT approach is contiguous pmt index array 
with category and qe_scale so can start from pmtid and get the pmtcat
and the qe for an energy.::

    jcv _PMTSimParamData
    ./Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/_PMTSimParamData.h

**/

#include "NPFold.h"

#ifdef WITH_JPMT
#include "JPMT.h"
#else
#include "SPMT.h"
#endif


const NPFold* get_jpmt_fold()
{
    const NPFold* pmt_f = nullptr ;  
#ifdef WITH_JPMT
    pmt_f = JPMT::Serialize(); 
#else
    pmt_f = SPMT::Serialize() ; 
#endif
    return pmt_f ; 
}




