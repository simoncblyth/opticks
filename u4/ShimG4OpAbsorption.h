#pragma once
#include "G4OpAbsorption.hh"
#include <csignal>
class ShimG4OpAbsorption : public G4OpAbsorption 
{
    public:
#ifdef DEBUG_TAG
        static const bool FLOAT ; 
        void ResetNumberOfInteractionLengthLeft(); 
        G4double GetMeanFreePath(const G4Track& aTrack,
                 G4double ,
                 G4ForceCondition* );

#endif
};

#ifdef DEBUG_TAG
/**
ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft
--------------------------------------------------------

Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify

**/

const bool ShimG4OpAbsorption::FLOAT = getenv("ShimG4OpAbsorption_FLOAT") != nullptr ;

//inline void ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); }
inline void ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft()
{
    //std::cout << "ShimG4OpAbsorption::FLOAT " << FLOAT << std::endl ; 
    theNumberOfInteractionLengthLeft =  FLOAT ? -1.f*std::log( float(G4UniformRand()) ): -1.*G4Log( G4UniformRand() ) ; 
    theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft; 
}

inline G4double ShimG4OpAbsorption::GetMeanFreePath(const G4Track& aTrack,  G4double, G4ForceCondition* )
{
     G4double len = G4OpAbsorption::GetMeanFreePath( aTrack, 0, 0 ); 
     //std::cout << "ShimG4OpAbsorption::GetMeanFreePath " << len << std::endl ; 

     std::raise(SIGINT); 

     return len ; 
}


#endif
 
