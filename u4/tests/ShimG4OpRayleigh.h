#pragma once
/**
ShimG4OpRayleigh 
===================

class G4OpRayleigh : public G4VDiscreteProcess

**/

#include "G4OpRayleigh.hh"
class ShimG4OpRayleigh : public G4OpRayleigh 
{
    public:
#ifdef DEBUG_TAG
        static const bool FLOAT ; 
        static const int  PIDX ; 
        void ResetNumberOfInteractionLengthLeft(); 
        G4double GetMeanFreePath(const G4Track& aTrack,
                 G4double ,
                 G4ForceCondition* );

      G4double PostStepGetPhysicalInteractionLength(
                             const G4Track& track,
                 G4double   previousStepSize,
                 G4ForceCondition* condition
                );  

#endif
}; 


