#pragma once
/**
U4Stack : classifying SBacktrace stack summaries used from U4Random::getFlatTag 
==================================================================================

Notice this makes the assumption that the backtrace of a U4Random::flat call 
is distinctive enough to identify what the random throw is being used for.
Multiple different uses of G4UniformRand within a single method could 
easily break this approach. 

It would then be necessary to develop some more complicated ways 
to identify the purpose of the random.  

CAUTION : trailing blanks in the stack summry literal strings could 
easily prevent matches. Use "set list" in vim to check. 
**/

#include <cassert>
#include <cstring>
#include "stag.h"

enum 
{
    U4Stack_Unclassified            = 0,
    U4Stack_RestDiscreteReset       = 1,
    U4Stack_DiscreteReset           = 2, 
    U4Stack_ScintDiscreteReset      = 3,
    U4Stack_BoundaryDiscreteReset   = 4,
    U4Stack_RayleighDiscreteReset   = 5,
    U4Stack_AbsorptionDiscreteReset = 6,
    U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb = 7,
    U4Stack_BoundaryDiDiTransCoeff  = 8,
    U4Stack_AbsorptionEffDetect     = 9,
    U4Stack_RayleighScatter         = 10,
    U4Stack_BoundaryDiMeReflectivity = 11
};

struct U4Stack 
{
    static unsigned Classify(const char* summary); 
    static bool IsClassified(unsigned stack); 
    static const char* Name(unsigned stack); 
    static unsigned Code(const char* name); 
    static unsigned TagToStack(unsigned tag); 

    static constexpr const char* Unclassified_      = "Unclassified" ;        // 0
    static constexpr const char* RestDiscreteReset_ = "RestDiscreteReset" ;   // 1: not used as use Shims to idenify processes in SBacktrace
    static constexpr const char* DiscreteReset_ = "DiscreteReset" ; // 2    : G4OpAbsoption G4OpRayleigh but now use Shims to identify 
    static constexpr const char* ScintDiscreteReset_ = "ScintDiscreteReset" ;  // 3
    static constexpr const char* BoundaryDiscreteReset_ = "BoundaryDiscreteReset" ; // 4
    static constexpr const char* RayleighDiscreteReset_ = "RayleighDiscreteReset" ;  // 5
    static constexpr const char* AbsorptionDiscreteReset_ = "AbsorptionDiscreteReset" ; // 6
    static constexpr const char* BoundaryBurn_SurfaceReflectTransmitAbsorb_ = "BoundaryBurn_SurfaceReflectTransmitAbsorb" ;  // 7
    static constexpr const char* BoundaryDiDiTransCoeff_ = "BoundaryDiDiTransCoeff" ;  // 8 
    static constexpr const char* AbsorptionEffDetect_ = "AbsorptionEffDetect" ;   // 9 
    static constexpr const char* RayleighScatter_ = "RayleighScatter" ;  // 10 
    static constexpr const char* BoundaryDiMeReflectivity_ = "BoundaryDiMeReflectivity" ;  // 11 





    static constexpr const char* RestDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* RestDiscreteReset_note = R"(
must be Scintillation, as only RestDiscrete process around
)" ; 


    static constexpr const char* DiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* ScintDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
DsG4Scintillation::ResetNumberOfInteractionLengthLeft
G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* BoundaryDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
InstrumentedG4OpBoundaryProcess::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* BoundaryDiscreteReset2_ = "BoundaryDiscreteReset2" ; // 4
    static constexpr const char* BoundaryDiscreteReset2 = R"(
U4Random::flat
InstrumentedG4OpBoundaryProcess::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 


    static constexpr const char* RayleighDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* ShimRayleighDiscreteReset_ = "ShimRayleighDiscreteReset" ;  // 5
    static constexpr const char* ShimRayleighDiscreteReset = R"(
U4Random::flat
ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* Shim2RayleighDiscreteReset_ = "Shim2RayleighDiscreteReset" ;  // 5
    static constexpr const char* Shim2RayleighDiscreteReset = R"(
U4Random::flat
ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft
ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 



    static constexpr const char* AbsorptionDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 
    static constexpr const char* ShimAbsorptionDiscreteReset_ = "ShimAbsorptionDiscreteReset" ; // 6
    static constexpr const char* ShimAbsorptionDiscreteReset = R"(
U4Random::flat
ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* Shim2AbsorptionDiscreteReset_ = "Shim2AbsorptionDiscreteReset" ; // 6
    static constexpr const char* Shim2AbsorptionDiscreteReset = R"(
U4Random::flat
ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft
ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 


    static constexpr const char* BoundaryBurn_SurfaceReflectTransmitAbsorb = R"(
U4Random::flat
InstrumentedG4OpBoundaryProcess::PostStepDoIt
G4SteppingManager::InvokePSDIP
G4SteppingManager::InvokePostStepDoItProcs
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* BoundaryBurn_SurfaceReflectTransmitAbsorb_note = R"(
BoundaryBurn_SurfaceReflectTransmitAbsorb : Chameleon Rand 
------------------------------------------------------------

BoundaryBurn
    when no surface is associated to a boundary this random does nothing : always a burn 

SurfaceReflectTransmitAbsorb
    when a surface with reflectivity less than 1. is associated this rand comes alive 
    and determines the reflect/absorb/transmit decision  

)" ; 


    static constexpr const char* BoundaryDiDiTransCoeff = R"(
U4Random::flat
InstrumentedG4OpBoundaryProcess::G4BooleanRand
InstrumentedG4OpBoundaryProcess::DielectricDielectric
InstrumentedG4OpBoundaryProcess::PostStepDoIt
G4SteppingManager::InvokePSDIP
G4SteppingManager::InvokePostStepDoItProcs
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* AbsorptionEffDetect = R"(
U4Random::flat
InstrumentedG4OpBoundaryProcess::G4BooleanRand
InstrumentedG4OpBoundaryProcess::DoAbsorption
InstrumentedG4OpBoundaryProcess::PostStepDoIt
G4SteppingManager::InvokePSDIP
G4SteppingManager::InvokePostStepDoItProcs
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* RayleighScatter = R"(
U4Random::flat
G4OpRayleigh::PostStepDoIt
G4SteppingManager::InvokePSDIP
G4SteppingManager::InvokePostStepDoItProcs
G4SteppingManager::Stepping
)" ; 


}; 

inline unsigned U4Stack::Classify(const char* summary)
{
    unsigned stack = U4Stack_Unclassified ; 
    if(strstr(summary, RestDiscreteReset))             stack = U4Stack_RestDiscreteReset ; 
    if(strstr(summary, DiscreteReset))                 stack = U4Stack_DiscreteReset ; 
    if(strstr(summary, ScintDiscreteReset))            stack = U4Stack_ScintDiscreteReset ; 
    if(strstr(summary, BoundaryDiscreteReset))         stack = U4Stack_BoundaryDiscreteReset ; 
    if(strstr(summary, BoundaryDiscreteReset2))        stack = U4Stack_BoundaryDiscreteReset ; 

    if(strstr(summary, RayleighDiscreteReset))         stack = U4Stack_RayleighDiscreteReset ; 
    if(strstr(summary, ShimRayleighDiscreteReset))     stack = U4Stack_RayleighDiscreteReset ; 
    if(strstr(summary, Shim2RayleighDiscreteReset))    stack = U4Stack_RayleighDiscreteReset ; 

    if(strstr(summary, AbsorptionDiscreteReset))       stack = U4Stack_AbsorptionDiscreteReset ; 
    if(strstr(summary, ShimAbsorptionDiscreteReset))   stack = U4Stack_AbsorptionDiscreteReset ; 
    if(strstr(summary, Shim2AbsorptionDiscreteReset))  stack = U4Stack_AbsorptionDiscreteReset ; 

    if(strstr(summary, BoundaryBurn_SurfaceReflectTransmitAbsorb)) stack = U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb ; 
    if(strstr(summary, BoundaryDiDiTransCoeff))        stack = U4Stack_BoundaryDiDiTransCoeff ; 
    if(strstr(summary, AbsorptionEffDetect))           stack = U4Stack_AbsorptionEffDetect ; 
    if(strstr(summary, RayleighScatter))               stack = U4Stack_RayleighScatter ; 
    return stack ; 
}
inline bool U4Stack::IsClassified(unsigned stack)
{
    return stack != U4Stack_Unclassified ; 
}

inline const char* U4Stack::Name(unsigned stack)
{
    const char* s = nullptr ; 
    switch(stack)
    {
        case U4Stack_Unclassified:                  s = Unclassified_            ; break ; 
        case U4Stack_RestDiscreteReset:             s = RestDiscreteReset_       ; break ; 
        case U4Stack_DiscreteReset:                 s = DiscreteReset_           ; break ; 
        case U4Stack_ScintDiscreteReset:            s = ScintDiscreteReset_      ; break ; 
        case U4Stack_BoundaryDiscreteReset:         s = BoundaryDiscreteReset_   ; break ; 
        case U4Stack_RayleighDiscreteReset:         s = RayleighDiscreteReset_   ; break ; 
        case U4Stack_AbsorptionDiscreteReset:       s = AbsorptionDiscreteReset_ ; break ; 
        case U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb: s = BoundaryBurn_SurfaceReflectTransmitAbsorb_  ; break ; 
        case U4Stack_BoundaryDiDiTransCoeff:        s = BoundaryDiDiTransCoeff_  ; break ; 
        case U4Stack_AbsorptionEffDetect:           s = AbsorptionEffDetect_     ; break ; 
        case U4Stack_RayleighScatter:               s = RayleighScatter_         ; break ; 
        case U4Stack_BoundaryDiMeReflectivity:      s = BoundaryDiMeReflectivity_ ; break ;
    }
    if(s) assert( Code(s) == stack ) ; 
    return s ; 
}

inline unsigned U4Stack::Code(const char* name)
{
    unsigned stack = U4Stack_Unclassified ; 
    if(strcmp(name, Unclassified_) == 0 )                  stack = U4Stack_Unclassified  ; 
    if(strcmp(name, RestDiscreteReset_) == 0 )             stack = U4Stack_RestDiscreteReset  ;
    if(strcmp(name, DiscreteReset_) == 0)                  stack = U4Stack_DiscreteReset ; 
    if(strcmp(name, ScintDiscreteReset_) == 0 )            stack = U4Stack_ScintDiscreteReset  ;
    if(strcmp(name, BoundaryDiscreteReset_) == 0)          stack = U4Stack_BoundaryDiscreteReset ; 

    if(strcmp(name, RayleighDiscreteReset_) == 0 )         stack = U4Stack_RayleighDiscreteReset  ;
    if(strcmp(name, ShimRayleighDiscreteReset_) == 0 )     stack = U4Stack_RayleighDiscreteReset  ;
    if(strcmp(name, Shim2RayleighDiscreteReset_) == 0 )    stack = U4Stack_RayleighDiscreteReset  ;

    if(strcmp(name, AbsorptionDiscreteReset_) == 0 )       stack = U4Stack_AbsorptionDiscreteReset  ;
    if(strcmp(name, ShimAbsorptionDiscreteReset_) == 0 )   stack = U4Stack_AbsorptionDiscreteReset  ;
    if(strcmp(name, Shim2AbsorptionDiscreteReset_) == 0 )  stack = U4Stack_AbsorptionDiscreteReset  ;

    if(strcmp(name, BoundaryBurn_SurfaceReflectTransmitAbsorb_) == 0)  stack = U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb ; 
    if(strcmp(name, BoundaryDiDiTransCoeff_) == 0)         stack = U4Stack_BoundaryDiDiTransCoeff ; 
    if(strcmp(name, AbsorptionEffDetect_) == 0)            stack = U4Stack_AbsorptionEffDetect ; 
    if(strcmp(name, RayleighScatter_) == 0 )               stack = U4Stack_RayleighScatter  ;
    if(strcmp(name, BoundaryDiMeReflectivity_) == 0 )      stack = U4Stack_BoundaryDiMeReflectivity  ;

    return stack ; 
}

/**
U4Stack::TagToStack
--------------------

Attempt at mapping from A:tag to B:stack 

* where to use this mapping anyhow ? unkeen to do this at C++ level as it feels like a complication 
  and potential info loss that is only not-info loss when are in an aligned state 

* but inevitably when generalize will get out of alignment and will need to use the A:tag  
  and B:stack to regain alignment 

* hence the right place to use the mapping is in python 

**/

inline unsigned U4Stack::TagToStack(unsigned tag)
{
    unsigned stack = U4Stack_Unclassified ;
    switch(tag)
    {
        case stag_undef:      stack = U4Stack_Unclassified                              ; break ;        // 0 -> 0
        case stag_to_sci:     stack = U4Stack_ScintDiscreteReset                        ; break ;        // 1 -> 3
        case stag_to_bnd:     stack = U4Stack_BoundaryDiscreteReset                     ; break ;        // 2 -> 4 
        case stag_to_sca:     stack = U4Stack_RayleighDiscreteReset                     ; break ;        // 3 -> 5 
        case stag_to_abs:     stack = U4Stack_AbsorptionDiscreteReset                   ; break ;        // 4 -> 6 
        case stag_at_burn_sf_sd:    stack = U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb ; break ;  // 5 -> 7 
        case stag_at_ref:     stack = U4Stack_BoundaryDiDiTransCoeff                    ; break ;        // 6 -> 8 
        case stag_sf_burn:    stack = U4Stack_AbsorptionEffDetect                       ; break ;        // 7 -> 9
        case stag_sc:         stack = U4Stack_RayleighScatter                           ; break ;        // 8 -> 10
        case stag_to_ree:     stack = U4Stack_Unclassified ; break ;  // 9
        case stag_re_wl:      stack = U4Stack_Unclassified ; break ;  // 10
        case stag_re_mom_ph:  stack = U4Stack_Unclassified ; break ;  // 11
        case stag_re_mom_ct:  stack = U4Stack_Unclassified ; break ;  // 12
        case stag_re_pol_ph:  stack = U4Stack_Unclassified ; break ;  // 13
        case stag_re_pol_ct:  stack = U4Stack_Unclassified ; break ;  // 14
        case stag_hp_ph:      stack = U4Stack_Unclassified ; break ;  // 15
        //case stag_hp_ct:      stack = U4Stack_Unclassified ; break ;  // 16 
    }
    return stack ; 
}

