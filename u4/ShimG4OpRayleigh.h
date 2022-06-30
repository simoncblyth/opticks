#pragma once
#include "G4OpRayleigh.hh"
class ShimG4OpRayleigh : public G4OpRayleigh 
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


const bool ShimG4OpRayleigh::FLOAT = getenv("ShimG4OpRayleigh_FLOAT") != nullptr ;

/**
ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft
-------------------------------------------------------

Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify


**/

//inline void ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); }
inline void ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft()
{
    //std::cout << "ShimG4OpRayleigh::FLOAT " << FLOAT << std::endl ; 
    theNumberOfInteractionLengthLeft =  FLOAT ? -1.f*std::log( float(G4UniformRand()) ): -1.*G4Log( G4UniformRand() ) ; 
    theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft; 
}


inline G4double ShimG4OpRayleigh::GetMeanFreePath(const G4Track& aTrack,  G4double, G4ForceCondition* )
{
     G4double len = G4OpRayleigh::GetMeanFreePath( aTrack, 0, 0 ); 
     //std::cout << "ShimG4OpRayleigh::GetMeanFreePath " << len << std::endl ; 
     return len ; 
}


