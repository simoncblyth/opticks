#pragma once
/**
U4Stack : classifying SBacktrace stack summaries used from U4Random::getFlatTag 
==================================================================================

CAUTION : trailing blanks in the literal strings, use "set list"

**/

#include <cassert>

enum 
{
    U4Stack_Unclassified            = 0,
    U4Stack_RestDiscreteReset       = 1,
    U4Stack_ScintDiscreteReset      = 2,
    U4Stack_AbsorptionDiscreteReset = 3,
    U4Stack_RayleighDiscreteReset   = 4,
    U4Stack_DiscreteReset           = 5, 
    U4Stack_BoundaryDiscreteReset   = 6,
    U4Stack_BoundaryDiDi            = 7,
    U4Stack_BoundaryBurn            = 8
};

struct U4Stack 
{
    static unsigned Classify(const char* summary); 
    static bool IsClassified(unsigned stack); 
    static const char* Name(unsigned stack); 
    static unsigned Code(const char* name); 

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

    static constexpr const char* BoundaryDiDi_ = "BoundaryDiDi" ; 
    static constexpr const char* BoundaryDiDi = R"(
U4Random::flat
InstrumentedG4OpBoundaryProcess::G4BooleanRand
InstrumentedG4OpBoundaryProcess::DielectricDielectric
InstrumentedG4OpBoundaryProcess::PostStepDoIt
G4SteppingManager::InvokePSDIP
G4SteppingManager::InvokePostStepDoItProcs
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* BoundaryBurn_ = "BoundaryBurn" ; 
    static constexpr const char* BoundaryBurn = R"(
U4Random::flat
InstrumentedG4OpBoundaryProcess::PostStepDoIt
G4SteppingManager::InvokePSDIP
G4SteppingManager::InvokePostStepDoItProcs
G4SteppingManager::Stepping
)" ; 

}; 

inline unsigned U4Stack::Classify(const char* summary)
{
    unsigned stack = U4Stack_Unclassified ; 
    if(strstr(summary, RestDiscreteReset))        stack = U4Stack_RestDiscreteReset ; 
    if(strstr(summary, ScintDiscreteReset))       stack = U4Stack_ScintDiscreteReset ; 
    if(strstr(summary, AbsorptionDiscreteReset))  stack = U4Stack_AbsorptionDiscreteReset ; 
    if(strstr(summary, RayleighDiscreteReset))    stack = U4Stack_RayleighDiscreteReset ; 
    if(strstr(summary, DiscreteReset))            stack = U4Stack_DiscreteReset ; 
    if(strstr(summary, BoundaryDiscreteReset))    stack = U4Stack_BoundaryDiscreteReset ; 
    if(strstr(summary, BoundaryDiDi))             stack = U4Stack_BoundaryDiDi ; 
    if(strstr(summary, BoundaryBurn))             stack = U4Stack_BoundaryBurn ; 
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
        case U4Stack_Unclassified:               s = Unclassified_            ; break ; 
        case U4Stack_RestDiscreteReset:          s = RestDiscreteReset_       ; break ; 
        case U4Stack_ScintDiscreteReset:         s = ScintDiscreteReset_      ; break ; 
        case U4Stack_AbsorptionDiscreteReset:    s = AbsorptionDiscreteReset_ ; break ; 
        case U4Stack_RayleighDiscreteReset:      s = RayleighDiscreteReset_   ; break ; 
        case U4Stack_DiscreteReset:              s = DiscreteReset_           ; break ; 
        case U4Stack_BoundaryDiscreteReset:      s = BoundaryDiscreteReset_   ; break ; 
        case U4Stack_BoundaryDiDi:               s = BoundaryDiDi_            ; break ; 
        case U4Stack_BoundaryBurn:               s = BoundaryBurn_            ; break ; 
    }
    if(s) assert( Code(s) == stack ) ; 
    return s ; 
}

inline unsigned U4Stack::Code(const char* name)
{
    unsigned stk = U4Stack_Unclassified ; 
    if(strcmp(name, Unclassified_) == 0 )            stk = U4Stack_Unclassified  ; 
    if(strcmp(name, RestDiscreteReset_) == 0 )       stk = U4Stack_RestDiscreteReset  ;
    if(strcmp(name, ScintDiscreteReset_) == 0 )      stk = U4Stack_ScintDiscreteReset  ;
    if(strcmp(name, AbsorptionDiscreteReset_) == 0 ) stk = U4Stack_AbsorptionDiscreteReset  ;
    if(strcmp(name, RayleighDiscreteReset_) == 0 )   stk = U4Stack_RayleighDiscreteReset  ;
    if(strcmp(name, DiscreteReset_) == 0)            stk = U4Stack_DiscreteReset ; 
    if(strcmp(name, BoundaryDiscreteReset_) == 0)    stk = U4Stack_BoundaryDiscreteReset ; 
    if(strcmp(name, BoundaryDiDi_) == 0)             stk = U4Stack_BoundaryDiDi ; 
    if(strcmp(name, BoundaryBurn_) == 0)             stk = U4Stack_BoundaryBurn ; 
    return stk ; 
}

