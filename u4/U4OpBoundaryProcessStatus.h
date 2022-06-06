#pragma once
/**
U4OpBoundaryProcessStatus
==========================

HMM duplicates X4OpBoundaryProcessStatus, usage::

    #include "G4OpBoundaryProcess.hh"
    #include "U4OpBoundaryProcessStatus.hh"

    const char* name = U4OpBoundaryProcessStatus::Name( FresnelRefraction );  

**/

struct U4OpBoundaryProcessStatus
{
    static const char* Name(unsigned status);

    static constexpr const char* Undefined_ = "Undefined" ;
    static constexpr const char* Transmission_ = "Transmission" ;
    static constexpr const char* FresnelRefraction_ = "FresnelRefraction" ;
    static constexpr const char* FresnelReflection_  = "FresnelReflection" ;;
    static constexpr const char* TotalInternalReflection_ = "TotalInternalReflection" ;
    static constexpr const char* LambertianReflection_ = "LambertianReflection" ;
    static constexpr const char* LobeReflection_ = "LobeReflection" ;
    static constexpr const char* SpikeReflection_ = "SpikeReflection" ;
    static constexpr const char* BackScattering_ = "BackScattering" ;
    static constexpr const char* Absorption_ = "Absorption" ;
    static constexpr const char* Detection_ = "Detection" ;
    static constexpr const char* NotAtBoundary_ = "NotAtBoundary" ;
    static constexpr const char* SameMaterial_ = "SameMaterial" ;
    static constexpr const char* StepTooSmall_ = "StepTooSmall" ;
    static constexpr const char* NoRINDEX_ = "NoRINDEX" ;
    static constexpr const char* Other_ = "Other" ;   
};


inline const char* U4OpBoundaryProcessStatus::Name(unsigned status)
{
    const char* s = nullptr ; 
    switch(status)
    {   
       case Undefined: s = Undefined_ ; break ; 
       case Transmission: s = Transmission_ ; break ; 
       case FresnelRefraction: s = FresnelRefraction_ ; break ; 
       case FresnelReflection: s = FresnelReflection_ ; break ; 
       case TotalInternalReflection: s = TotalInternalReflection_ ; break ; 
       case LambertianReflection: s = LambertianReflection_ ; break ; 
       case LobeReflection: s = LobeReflection_ ; break ; 
       case SpikeReflection: s = SpikeReflection_ ; break ; 
       case BackScattering: s = BackScattering_ ; break ; 
       case Absorption: s = Absorption_ ; break ; 
       case Detection: s = Detection_ ; break ; 
       case NotAtBoundary: s = NotAtBoundary_ ; break ; 
       case SameMaterial: s = SameMaterial_ ; break ; 
       case StepTooSmall: s = StepTooSmall_ ; break ; 
       case NoRINDEX: s = NoRINDEX_ ; break ; 
       default: s = Other_ ; break ; 
    }   
    return s ; 
}

