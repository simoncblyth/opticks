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
#include "stag.h"

enum 
{
    U4Stack_Unclassified            = 0,
    U4Stack_RestDiscreteReset       = 1,
    U4Stack_ScintDiscreteReset      = 2,
    U4Stack_AbsorptionDiscreteReset = 3,
    U4Stack_RayleighDiscreteReset   = 4,
    U4Stack_DiscreteReset           = 5, 
    U4Stack_BoundaryDiscreteReset   = 6,
    U4Stack_BoundaryDiDiTransCoeff  = 7,
    U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb = 8,
    U4Stack_AbsorptionEffDetect     = 9
};

struct U4Stack 
{
    static unsigned Classify(const char* summary); 
    static bool IsClassified(unsigned stack); 
    static const char* Name(unsigned stack); 
    static unsigned Code(const char* name); 
    static unsigned TagToStack(unsigned tag); 

    static constexpr const char* Unclassified_      = "Unclassified" ; 
    static constexpr const char* RestDiscreteReset_ = "RestDiscreteReset" ;   // must be Scintillation, as only RestDiscrete process around 
    static constexpr const char* RestDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 
    
    static constexpr const char* ScintDiscreteReset_ = "ScintDiscreteReset" ;
    static constexpr const char* ScintDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
DsG4Scintillation::ResetNumberOfInteractionLengthLeft
G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* AbsorptionDiscreteReset_ = "AbsorptionDiscreteReset" ;
    static constexpr const char* AbsorptionDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* RayleighDiscreteReset_ = "RayleighDiscreteReset" ;
    static constexpr const char* RayleighDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* DiscreteReset_ = "DiscreteReset" ; // G4OpAbsoption G4OpRayleigh
    static constexpr const char* DiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* BoundaryDiscreteReset_ = "BoundaryDiscreteReset" ; 
    static constexpr const char* BoundaryDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
InstrumentedG4OpBoundaryProcess::ResetNumberOfInteractionLengthLeft
G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* BoundaryDiDiTransCoeff_ = "BoundaryDiDiTransCoeff" ; 
    static constexpr const char* BoundaryDiDiTransCoeff = R"(
U4Random::flat
InstrumentedG4OpBoundaryProcess::G4BooleanRand
InstrumentedG4OpBoundaryProcess::DielectricDielectric
InstrumentedG4OpBoundaryProcess::PostStepDoIt
G4SteppingManager::InvokePSDIP
G4SteppingManager::InvokePostStepDoItProcs
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* BoundaryBurn_SurfaceReflectTransmitAbsorb_ = "BoundaryBurn_SurfaceReflectTransmitAbsorb" ; 
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


    static constexpr const char* AbsorptionEffDetect_ = "AbsorptionEffDetect" ; 
    static constexpr const char* AbsorptionEffDetect = R"(
U4Random::flat
InstrumentedG4OpBoundaryProcess::G4BooleanRand
InstrumentedG4OpBoundaryProcess::DoAbsorption
InstrumentedG4OpBoundaryProcess::PostStepDoIt
G4SteppingManager::InvokePSDIP
G4SteppingManager::InvokePostStepDoItProcs
G4SteppingManager::Stepping
)" ; 

}; 

inline unsigned U4Stack::Classify(const char* summary)
{
    unsigned stack = U4Stack_Unclassified ; 
    if(strstr(summary, RestDiscreteReset))             stack = U4Stack_RestDiscreteReset ; 
    if(strstr(summary, ScintDiscreteReset))            stack = U4Stack_ScintDiscreteReset ; 
    if(strstr(summary, AbsorptionDiscreteReset))       stack = U4Stack_AbsorptionDiscreteReset ; 
    if(strstr(summary, RayleighDiscreteReset))         stack = U4Stack_RayleighDiscreteReset ; 
    if(strstr(summary, DiscreteReset))                 stack = U4Stack_DiscreteReset ; 
    if(strstr(summary, BoundaryDiscreteReset))         stack = U4Stack_BoundaryDiscreteReset ; 
    if(strstr(summary, BoundaryDiDiTransCoeff))        stack = U4Stack_BoundaryDiDiTransCoeff ; 
    if(strstr(summary, BoundaryBurn_SurfaceReflectTransmitAbsorb)) stack = U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb ; 
    if(strstr(summary, AbsorptionEffDetect))           stack = U4Stack_AbsorptionEffDetect ; 
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
        case U4Stack_ScintDiscreteReset:            s = ScintDiscreteReset_      ; break ; 
        case U4Stack_AbsorptionDiscreteReset:       s = AbsorptionDiscreteReset_ ; break ; 
        case U4Stack_RayleighDiscreteReset:         s = RayleighDiscreteReset_   ; break ; 
        case U4Stack_DiscreteReset:                 s = DiscreteReset_           ; break ; 
        case U4Stack_BoundaryDiscreteReset:         s = BoundaryDiscreteReset_   ; break ; 
        case U4Stack_BoundaryDiDiTransCoeff:        s = BoundaryDiDiTransCoeff_  ; break ; 
        case U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb: s = BoundaryBurn_SurfaceReflectTransmitAbsorb_  ; break ; 
        case U4Stack_AbsorptionEffDetect:           s = AbsorptionEffDetect_            ; break ; 
    }
    if(s) assert( Code(s) == stack ) ; 
    return s ; 
}

inline unsigned U4Stack::Code(const char* name)
{
    unsigned stack = U4Stack_Unclassified ; 
    if(strcmp(name, Unclassified_) == 0 )                  stack = U4Stack_Unclassified  ; 
    if(strcmp(name, RestDiscreteReset_) == 0 )             stack = U4Stack_RestDiscreteReset  ;
    if(strcmp(name, ScintDiscreteReset_) == 0 )            stack = U4Stack_ScintDiscreteReset  ;
    if(strcmp(name, AbsorptionDiscreteReset_) == 0 )       stack = U4Stack_AbsorptionDiscreteReset  ;
    if(strcmp(name, RayleighDiscreteReset_) == 0 )         stack = U4Stack_RayleighDiscreteReset  ;
    if(strcmp(name, DiscreteReset_) == 0)                  stack = U4Stack_DiscreteReset ; 
    if(strcmp(name, BoundaryDiscreteReset_) == 0)          stack = U4Stack_BoundaryDiscreteReset ; 
    if(strcmp(name, BoundaryDiDiTransCoeff_) == 0)         stack = U4Stack_BoundaryDiDiTransCoeff ; 
    if(strcmp(name, BoundaryBurn_SurfaceReflectTransmitAbsorb_) == 0)  stack = U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb ; 
    if(strcmp(name, AbsorptionEffDetect_) == 0)            stack = U4Stack_AbsorptionEffDetect ; 
    return stack ; 
}

/**
U4Stack::TagToStack
--------------------

Attempt at mapping from A:tag to B:stack 

**/

inline unsigned U4Stack::TagToStack(unsigned tag)
{
    unsigned stack = U4Stack_Unclassified ;
    switch(tag)
    {
        case stag_undef:      stack = U4Stack_Unclassified                              ; break ;  // 0 -> 0
        case stag_to_sci:     stack = U4Stack_ScintDiscreteReset                        ; break ;  // 1 -> 2
        case stag_to_bnd:     stack = U4Stack_BoundaryDiscreteReset                     ; break ;  // 2 -> 6 
        case stag_to_sca:     stack = U4Stack_RayleighDiscreteReset                     ; break ;  // 3 -> 4 
        case stag_to_abs:     stack = U4Stack_AbsorptionDiscreteReset                   ; break ;  // 4 -> 3 
        case stag_at_burn:    stack = U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb ; break ;  // 5 -> 8 
        case stag_at_ref:     stack = U4Stack_BoundaryDiDiTransCoeff                    ; break ;  // 6 -> 7 
        case stag_sf_sd:      stack = U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb ; break ;  // 7 -> 8 
        case stag_sf_burn:    stack = U4Stack_AbsorptionEffDetect                       ; break ;  // 8 -> 9
        case stag_to_ree:     stack = U4Stack_Unclassified ; break ;  // 9
        case stag_re_wl:      stack = U4Stack_Unclassified ; break ;  // 10
        case stag_re_mom_ph:  stack = U4Stack_Unclassified ; break ;  // 11
        case stag_re_mom_ct:  stack = U4Stack_Unclassified ; break ;  // 12
        case stag_re_pol_ph:  stack = U4Stack_Unclassified ; break ;  // 13
        case stag_re_pol_ct:  stack = U4Stack_Unclassified ; break ;  // 14
        case stag_hp_ph:      stack = U4Stack_Unclassified ; break ;  // 15
        case stag_hp_ct:      stack = U4Stack_Unclassified ; break ;  // 16 
        case stag_sc_u0:      stack = U4Stack_Unclassified ; break ;  // 17
        case stag_sc_u1:      stack = U4Stack_Unclassified ; break ;  // 18
        case stag_sc_u2:      stack = U4Stack_Unclassified ; break ;  // 19
        case stag_sc_u3:      stack = U4Stack_Unclassified ; break ;  // 20
        case stag_sc_u4:      stack = U4Stack_Unclassified ; break ;  // 21
        case stag_br_align_0: stack = U4Stack_Unclassified ; break ;  // 22
        case stag_br_align_1: stack = U4Stack_BoundaryDiscreteReset    ; break ;  // 23 -> 6
        case stag_br_align_2: stack = U4Stack_RayleighDiscreteReset    ; break ;  // 24 -> 4 
        case stag_br_align_3: stack = U4Stack_AbsorptionDiscreteReset  ; break ;  // 25 -> 3
    }
    return stack ; 
}
