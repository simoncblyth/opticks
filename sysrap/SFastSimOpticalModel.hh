#pragma once
/**
SFastSimOpticalModel
======================

HMM: what about when there are multiple instances of 
FastSim in operation at same time, eg with one for each PMT type ? 

Cannot then assume a single INSTANCE. That will cause
overwriting of the INSTANCE : so the last one will win. 

Could keep updating the INSTANCE in the DoIt ? 

Need better way to get the FastSim status into U4Recorder,  
presumably using the name to distingish::

    1106 void
    1107 HamamatsuR12860PMTManager::helper_fast_sim()
    1108 {
    1114     pmtOpticalModel = new junoPMTOpticalModel(GetName()+"_optical_model",
    1115                                                                    body_phys, body_region);
    1116 


**/
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SFastSimOpticalModel
{
    static const SFastSimOpticalModel* Get() ; 
    static const SFastSimOpticalModel* INSTANCE ; 
    static char GetStatus() ; 

    const char* name ; 

    SFastSimOpticalModel(const char* name); 
 
    virtual char getStatus() const = 0 ;  // 'A' 'R' 'T' 'D' '?'
};





