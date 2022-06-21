#pragma once
/**
U4Stack : classifying SBacktrace stack summaries used from U4Random::getFlatTag 
==================================================================================

CAUTION : trailing blanks in the literal strings, use "set list"

**/

enum 
{
    U4Stack_Unclassified = 0,
    U4Stack_RestDiscreteReset = 1,
    U4Stack_DiscreteReset = 2, 
    U4Stack_BoundaryDiDi = 3,
    U4Stack_BoundaryBurn = 4
};

struct U4Stack 
{
    static unsigned Classify(const char* summary); 
    static bool IsClassified(unsigned stk); 
    static const char* Name(unsigned stk); 

    static constexpr const char* Unclassified_      = "Unclassified" ; 
    static constexpr const char* RestDiscreteReset_ = "RestDiscreteReset" ; 
    static constexpr const char* RestDiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
G4VProcess::PostStepGPIL
G4SteppingManager::DefinePhysicalStepLength
G4SteppingManager::Stepping
)" ; 

    static constexpr const char* DiscreteReset_ = "DiscreteReset" ; 
    static constexpr const char* DiscreteReset = R"(
U4Random::flat
G4VProcess::ResetNumberOfInteractionLengthLeft
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
    if(strstr(summary, DiscreteReset)     != nullptr ) stk = U4Stack_DiscreteReset ; 
    if(strstr(summary, RestDiscreteReset) != nullptr ) stk = U4Stack_RestDiscreteReset ; 
    if(strstr(summary, BoundaryDiDi)      != nullptr ) stk = U4Stack_BoundaryDiDi ; 
    if(strstr(summary, BoundaryBurn)      != nullptr ) stk = U4Stack_BoundaryBurn ; 
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
        case U4Stack_Unclassified:      s = Unclassified_      ; break ; 
        case U4Stack_RestDiscreteReset: s = RestDiscreteReset_ ; break ; 
        case U4Stack_DiscreteReset:     s = DiscreteReset_     ; break ; 
        case U4Stack_BoundaryDiDi:      s = BoundaryDiDi_     ; break ; 
        case U4Stack_BoundaryBurn:      s = BoundaryBurn_     ; break ; 
    }
    return s ; 
}


