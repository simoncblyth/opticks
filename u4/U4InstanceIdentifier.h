#pragma once
/**
U4InstanceIdentifier.h
=======================

Pure virtual protocol base used to interface Opticks geometry translation 
with detector specific code. 

getInstanceId
    method is called on the outer volume of every factorized instance during geometry translation, 
    the returned unsigned value is used by IAS_Builder to set the OptixInstance .instanceId 
    Within CSGOptiX/CSGOptiX7.cu:: __closesthit__ch *optixGetInstanceId()* is used to 
    passes the instanceId value into "quad2* prd" (per-ray-data) which is available 
    within qudarap/qsim.h methods. 
    
    The 32 bit unsigned returned by *getInstanceIdentity* may not use the top bit as ~0u 
    is reserved by OptiX to mean "not-an-instance". So this provides 31 bits of identity 
    information per instance.  

**/
class G4PVPlacement ; 

struct U4InstanceIdentifier
{
    virtual unsigned getInstanceId(const G4PVPlacement* pv) = 0 ; 
}; 


