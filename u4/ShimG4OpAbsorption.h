#pragma once

#include "G4OpAbsorption.hh"

class ShimG4OpAbsorption : public G4OpAbsorption 
{
    public:
#ifdef DEBUG_TAG
        // Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify
        void ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); } 
#endif

}; 
