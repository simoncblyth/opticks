#pragma once

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
    virtual char getCustomBoundaryStatus() const = 0 ; 

}; 


