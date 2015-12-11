#ifndef LXeSteppingVerbose_h
#define LXeSteppingVerbose_h 1

#include "G4SteppingVerbose.hh"

class LXeSteppingVerbose : public G4SteppingVerbose
{
  public:

    LXeSteppingVerbose();
    virtual ~LXeSteppingVerbose();

    virtual void StepInfo();
    virtual void TrackingStarted();

};

#endif
