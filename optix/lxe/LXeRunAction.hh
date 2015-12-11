#include "G4UserRunAction.hh"

#ifndef LXeRunAction_h
#define LXeRunAction_h 1

class LXeRecorderBase;

class LXeRunAction : public G4UserRunAction
{
  public:

    LXeRunAction(LXeRecorderBase*);
    virtual ~LXeRunAction();

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);

  private:

    LXeRecorderBase* fRecorder;
};

#endif
