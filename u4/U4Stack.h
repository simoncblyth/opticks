#pragma once
/**
U4Stack : classifying SBacktrace stack summaries used from U4Random::getFlatTag 
==================================================================================

CAUTION : trailing blanks in the literal strings, use "set list"

**/

#include <cassert>

enum 
{
    U4Stack_Unclassified          = 0,
    U4Stack_RestDiscreteReset     = 1,
    U4Stack_ScintDiscreteReset    = 2,
    U4Stack_DiscreteReset         = 3, 
    U4Stack_BoundaryDiscreteReset = 4,
    U4Stack_BoundaryDiDi          = 5,
    U4Stack_BoundaryBurn          = 6
};

struct U4Stack 
{
    static unsigned Classify(const char* summary); 
    static bool IsClassified(unsigned stk); 
    static const char* Name(unsigned stk); 
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
    unsigned stk = U4Stack_Unclassified ; 
    if(strstr(summary, RestDiscreteReset) != nullptr )     stk = U4Stack_RestDiscreteReset ; 
    if(strstr(summary, ScintDiscreteReset) != nullptr )    stk = U4Stack_ScintDiscreteReset ; 
    if(strstr(summary, DiscreteReset)     != nullptr )     stk = U4Stack_DiscreteReset ; 
    if(strstr(summary, BoundaryDiscreteReset) != nullptr ) stk = U4Stack_BoundaryDiscreteReset ; 
    if(strstr(summary, BoundaryDiDi)      != nullptr )     stk = U4Stack_BoundaryDiDi ; 
    if(strstr(summary, BoundaryBurn)      != nullptr )     stk = U4Stack_BoundaryBurn ; 
    return stk ; 
}
inline bool U4Stack::IsClassified(unsigned stk)
{
    return stk != U4Stack_Unclassified ; 
}

inline const char* U4Stack::Name(unsigned stk)
{
    const char* s = nullptr ; 
    switch(stk)
    {
        case U4Stack_Unclassified:          s = Unclassified_      ; break ; 
        case U4Stack_RestDiscreteReset:     s = RestDiscreteReset_ ; break ; 
        case U4Stack_ScintDiscreteReset:    s = ScintDiscreteReset_ ; break ; 
        case U4Stack_DiscreteReset:         s = DiscreteReset_     ; break ; 
        case U4Stack_BoundaryDiscreteReset: s = BoundaryDiscreteReset_ ; break ; 
        case U4Stack_BoundaryDiDi:          s = BoundaryDiDi_     ; break ; 
        case U4Stack_BoundaryBurn:          s = BoundaryBurn_     ; break ; 
    }
    if(s) assert( Code(s) == stk ) ; 
    return s ; 
}

inline unsigned U4Stack::Code(const char* name)
{
    unsigned stk = U4Stack_Unclassified ; 
    if(strcmp(name, Unclassified_) == 0 )         stk = U4Stack_Unclassified  ; 
    if(strcmp(name, RestDiscreteReset_) == 0 )    stk = U4Stack_RestDiscreteReset  ;
    if(strcmp(name, ScintDiscreteReset_) == 0 )   stk = U4Stack_ScintDiscreteReset  ;
    if(strcmp(name, DiscreteReset_) == 0)         stk = U4Stack_DiscreteReset ; 
    if(strcmp(name, BoundaryDiscreteReset_) == 0) stk = U4Stack_BoundaryDiscreteReset ; 
    if(strcmp(name, BoundaryDiDi_) == 0)          stk = U4Stack_BoundaryDiDi ; 
    if(strcmp(name, BoundaryBurn_) == 0)          stk = U4Stack_BoundaryBurn ; 
    return stk ; 
}

