#pragma once
#include "G4OpAbsorption.hh"
const bool ShimG4OpAbsorption_FLOAT = getenv("ShimG4OpAbsorption_FLOAT") != nullptr ; 
class ShimG4OpAbsorption : public G4OpAbsorption 
{
    public:
#ifdef DEBUG_TAG
        // Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify
        //void ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); } 
        void ResetNumberOfInteractionLengthLeft()
        {
            theNumberOfInteractionLengthLeft =  ShimG4OpAbsorption_FLOAT ? -1.f*std::log( float(G4UniformRand()) ): -1.*G4Log( G4UniformRand() ) ; 
            theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft; 
        }
#endif
}; 
