#pragma once

#include "G4OpRayleigh.hh"

class ShimG4OpRayleigh : public G4OpRayleigh 
{
    public:
#ifdef DEBUG_TAG
        // Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify
        void ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); } 
#endif

}; 

