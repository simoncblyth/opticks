#pragma once

#include "U4Stack.h"

struct U4StackAuto
{
    static unsigned Classify(const char* summary); 
    static bool IsClassified(unsigned stack); 

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



inline unsigned U4StackAuto::Classify(const char* summary)
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

inline bool U4StackAuto::IsClassified(unsigned stack)
{
    return stack != U4Stack_Unclassified ; 
}



