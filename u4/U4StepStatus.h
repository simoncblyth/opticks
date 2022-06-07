#pragma once

struct U4StepStatus
{
    static const char* Name(unsigned status); 

    static constexpr const char* fWorldBoundary_ = "fWorldBoundary" ;  
    static constexpr const char* fGeomBoundary_  = "fGeomBoundary" ;  
    static constexpr const char* fAtRestDoItProc_ = "fAtRestDoItProc" ; 
    static constexpr const char* fAlongStepDoItProc_ = "fAlongStepDoItProc" ; 
    static constexpr const char* fPostStepDoItProc_ = "fPostStepDoItProc" ; 
    static constexpr const char* fUserDefinedLimit_ = "fUserDefinedLimit" ; 
    static constexpr const char* fExclusivelyForcedProc_ = "fExclusivelyForcedProc" ; 
    static constexpr const char* fUndefined_ = "fUndefined" ; 
    static constexpr const char* fERROR_ = "fERROR" ; 
}; 

inline const char* U4StepStatus::Name(unsigned status)
{
    const char* s = nullptr ; 
    switch(status)
    {   
        case fWorldBoundary:          s=fWorldBoundary_          ;break; 
        case fGeomBoundary:           s=fGeomBoundary_           ;break; 
        case fAtRestDoItProc:         s=fAtRestDoItProc_         ;break; 
        case fAlongStepDoItProc:      s=fAlongStepDoItProc_      ;break; 
        case fPostStepDoItProc:       s=fPostStepDoItProc_       ;break; 
        case fUserDefinedLimit:       s=fUserDefinedLimit_       ;break; 
        case fExclusivelyForcedProc:  s=fExclusivelyForcedProc_  ;break; 
        case fUndefined:              s=fUndefined_              ;break; 
        default:                      s=fERROR_                  ;break;
    }   
    return s ; 
}

