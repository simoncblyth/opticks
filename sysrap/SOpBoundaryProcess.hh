#pragma once
/**
SOpBoundaryProcess.hh
=======================

Used from U4Recorder::UserSteppingAction_Optical

* within WITH_PMTFASTSIM InstrumentedG4OpBoundaryProcess ISA SOpBoundaryProcess
* this uses singleton INSTANCE as backdoor access to the BoundaryProcess

**/


#include "SYSRAP_API_EXPORT.hh"
struct SYSRAP_API SOpBoundaryProcess
{
    static SOpBoundaryProcess* INSTANCE ; 
    static SOpBoundaryProcess* Get() ;  

    SOpBoundaryProcess(const char* name); 
    const char* name ; 

    virtual double getU0() const = 0 ; 
    virtual int    getU0_idx() const = 0 ; 
    virtual const double* getRecoveredNormal() const = 0 ;
    virtual char getCustomStatus() const = 0 ; 

}; 


