#pragma once

#include "G4SteppingVerbose.hh"
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API SteppingVerbose : public G4SteppingVerbose
{
 public:
   SteppingVerbose();
   virtual ~SteppingVerbose();

   virtual void StepInfo();
   virtual void TrackingStarted();

};
#include "CFG4_TAIL.hh"

