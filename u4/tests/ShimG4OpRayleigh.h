#pragma once
/**
ShimG4OpRayleigh 
===================

class G4OpRayleigh : public G4VDiscreteProcess

**/

#include "G4OpRayleigh.hh"
#include "U4_API_EXPORT.hh"

class U4_API ShimG4OpRayleigh : public G4OpRayleigh 
{
    public:
        ShimG4OpRayleigh(); 
        virtual ~ShimG4OpRayleigh(); 


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


     G4VParticleChange* PostStepDoIt(const G4Track& aTrack,
                                     const G4Step&  aStep);




#endif
}; 


