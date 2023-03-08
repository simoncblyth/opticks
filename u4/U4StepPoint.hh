#pragma once

struct sphoton ; 
class G4StepPoint ; 

#include <string>
#include "plog/Severity.h"
#include "U4_API_EXPORT.hh"

enum { 
   U4StepPoint_Undefined, 
   U4StepPoint_NoProc, 
   U4StepPoint_Transportation, 
   U4StepPoint_OpRayleigh, 
   U4StepPoint_OpAbsorption, 
   U4StepPoint_OpFastSim, 
   U4StepPoint_OTHER 
};  

struct U4_API U4StepPoint
{
    static const plog::Severity LEVEL ; 
    static void Update(sphoton& photon, const G4StepPoint* point);
    static std::string DescPositionTime(const G4StepPoint* point ); 

    static constexpr const char* Undefined_      = "Undefined" ; 
    static constexpr const char* NoProc_         = "NoProc" ; 
    static constexpr const char* Transportation_ = "Transportation" ; 
    static constexpr const char* OpRayleigh_     = "OpRayleigh" ; 
    static constexpr const char* OpAbsorption_   = "OpAbsorption" ; 
    static constexpr const char* OTHER_          = "OTHER" ; 
    static           const char* OpFastSim_ ;  // default "fast_sim_man" configurable by envvar U4StepPoint_OpFastSim 

    static const char* ProcessName(const G4StepPoint* point); 
    static unsigned ProcessDefinedStepType(const G4StepPoint* point); 
    static unsigned ProcessDefinedStepType(const char* name); 
    static const char* ProcessDefinedStepTypeName(unsigned type); 

    static unsigned BoundaryFlag(unsigned status) ; 


    template <typename T>
    static bool IsTransportationBoundary(const G4StepPoint* point);  

    template <typename T>
    static unsigned Flag(const G4StepPoint* point, bool warn=true ); 


    template <typename T>
    static std::string Desc(const G4StepPoint* point); 

}; 


