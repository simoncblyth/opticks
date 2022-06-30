#pragma once
#include "G4OpRayleigh.hh"
const bool ShimG4OpRayleigh_FLOAT = getenv("ShimG4OpRayleigh_FLOAT") != nullptr ; 
class ShimG4OpRayleigh : public G4OpRayleigh 
{
    public:
#ifdef DEBUG_TAG
        // Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify
        //void ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); } 
        void ResetNumberOfInteractionLengthLeft()
        {
            theNumberOfInteractionLengthLeft =  ShimG4OpRayleigh_FLOAT ? -1.f*std::log( float(G4UniformRand()) ): -1.*G4Log( G4UniformRand() ) ; 
            theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft; 
        }
#endif
}; 
