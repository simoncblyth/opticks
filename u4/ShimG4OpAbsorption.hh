#pragma once
/**
ShimG4OpAbsorption
====================

G4OpAbsorption : public G4VDiscreteProcess

**/

#include "plog/Severity.h"
#include "G4OpAbsorption.hh"
#include "U4_API_EXPORT.hh"

class U4_API ShimG4OpAbsorption : public G4OpAbsorption 
{

        static const plog::Severity LEVEL ; 
    public:
        ShimG4OpAbsorption(const G4String& processName = "OpAbsorption",
                                G4ProcessType type = fOptical);
       ~ShimG4OpAbsorption();

        static const bool FLOAT ; 
        static const int  PIDX ; 
        static const bool PIDX_ENABLED ;
 
        void ResetNumberOfInteractionLengthLeft(); 
        G4double GetMeanFreePath(const G4Track& aTrack,
                 G4double ,
                 G4ForceCondition* );

      G4double PostStepGetPhysicalInteractionLength(
                             const G4Track& track,
                 G4double   previousStepSize,
                 G4ForceCondition* condition
                );  

};

