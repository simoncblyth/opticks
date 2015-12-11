#pragma once
#include "G4SteppingVerbose.hh"
class SteppingVerbose : public G4SteppingVerbose
{
 public:
   SteppingVerbose();
   virtual ~SteppingVerbose();

   virtual void StepInfo();
   virtual void TrackingStarted();

};

