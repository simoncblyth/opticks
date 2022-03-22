#pragma once
/**
X4OpBoundaryProcessStatus
===========================

Usage::

    #include "G4OpBoundaryProcess.hh"
    #include "X4OpBoundaryProcessStatus.hh"

    const char* name = X4OpBoundaryProcessStatus::Name( FresnelRefraction );  

**/
struct X4OpBoundaryProcessStatus
{
    static const char* Name(unsigned status);  

    static const char* Undefined_ ; 
    static const char* Transmission_ ; 
    static const char* FresnelRefraction_ ; 
    static const char* FresnelReflection_ ; 
    static const char* TotalInternalReflection_ ;
    static const char* LambertianReflection_ ; 
    static const char* LobeReflection_ ; 
    static const char* SpikeReflection_ ;
    static const char* BackScattering_ ;
    static const char* Absorption_ ; 
    static const char* Detection_ ;
    static const char* NotAtBoundary_ ;
    static const char* SameMaterial_ ; 
    static const char* StepTooSmall_ ; 
    static const char* NoRINDEX_ ;
    static const char* Other_ ;  
};

const char* X4OpBoundaryProcessStatus::Undefined_ = "Undefined" ; 
const char* X4OpBoundaryProcessStatus::Transmission_ = "Transmission" ; 
const char* X4OpBoundaryProcessStatus::FresnelRefraction_ = "FresnelRefraction" ; 
const char* X4OpBoundaryProcessStatus::FresnelReflection_ = "FresnelReflection" ; 
const char* X4OpBoundaryProcessStatus::TotalInternalReflection_ = "TotalInternalReflection" ; 
const char* X4OpBoundaryProcessStatus::LambertianReflection_ = "LambertianReflection" ; 
const char* X4OpBoundaryProcessStatus::LobeReflection_ = "LobeReflection" ; 
const char* X4OpBoundaryProcessStatus::SpikeReflection_ = "SpikeReflection" ; 
const char* X4OpBoundaryProcessStatus::BackScattering_ = "BackScattering" ; 
const char* X4OpBoundaryProcessStatus::Absorption_ = "Absorption" ; 
const char* X4OpBoundaryProcessStatus::Detection_ = "Detection" ; 
const char* X4OpBoundaryProcessStatus::NotAtBoundary_ = "NotAtBoundary" ; 
const char* X4OpBoundaryProcessStatus::SameMaterial_ = "SameMaterial" ; 
const char* X4OpBoundaryProcessStatus::StepTooSmall_ = "StepTooSmall" ; 
const char* X4OpBoundaryProcessStatus::NoRINDEX_ = "NoRINDEX" ; 
const char* X4OpBoundaryProcessStatus::Other_ = "Other" ; 

const char* X4OpBoundaryProcessStatus::Name(unsigned status)
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

