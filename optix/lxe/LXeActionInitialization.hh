#ifndef LXeActionInitialization_h
#define LXeActionInitialization_h 1

#include "G4VUserActionInitialization.hh"

class LXeRecorderBase;

/// Action initialization class.
///

class LXeActionInitialization : public G4VUserActionInitialization
{
  public:
    LXeActionInitialization(LXeRecorderBase*);
    virtual ~LXeActionInitialization();

    virtual void BuildForMaster() const;
    virtual void Build() const;

    virtual G4VSteppingVerbose* InitializeSteppingVerbose() const;

  private:
    LXeRecorderBase* fRecorder;
};

#endif
