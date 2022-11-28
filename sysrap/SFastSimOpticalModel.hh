#pragma once
/**
SFastSimOpticalModel
======================

This provided a kludge method to pass a status char 'A/R/T/D' 
from the PMTFastSim junoPMTOpticalModel::DoIt into U4StepPoint 
for use by U4Recorder.  But as this approach is limited to 
single PMT tests only, it was replaced by lodging information 
inside the trackinfo.  

When multiple PMTs are in use cannot then assume a single INSTANCE, 
so this approach would suffer from overwriting and confusion between the
status of multiple PMTs. 

::

    epsilon:opticks blyth$ opticks-fl SFastSimOpticalModel.hh
    ./sysrap/CMakeLists.txt
    ./sysrap/SFastSimOpticalModel.cc
    ./u4/U4StepPoint.cc
    epsilon:opticks blyth$ 


**/
#include "SYSRAP_API_EXPORT.hh"
#include <vector>

struct SYSRAP_API SFastSimOpticalModel
{
    static std::vector<SFastSimOpticalModel*> record ; 
    const char* name ; 
    SFastSimOpticalModel(const char* name); 
 
};





