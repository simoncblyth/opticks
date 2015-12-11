#ifndef LXeEventAction_h
#define LXeEventAction_h 1

#include "LXeEventMessenger.hh"
#include "G4UserEventAction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"

class G4Event;
class LXeRecorderBase;

class LXeEventAction : public G4UserEventAction
{
  public:

    LXeEventAction(LXeRecorderBase*);
    virtual ~LXeEventAction();

  public:

    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);

    void SetSaveThreshold(G4int );

    void SetEventVerbose(G4int v){fVerbose=v;}

    void SetPMTThreshold(G4int t){fPMTThreshold=t;}

    void SetForceDrawPhotons(G4bool b){fForcedrawphotons=b;}
    void SetForceDrawNoPhotons(G4bool b){fForcenophotons=b;}

  private:

    LXeRecorderBase* fRecorder;
    LXeEventMessenger* fEventMessenger;

    G4int              fSaveThreshold;

    G4int              fScintCollID;
    G4int              fPMTCollID;

    G4int              fVerbose;

    G4int              fPMTThreshold;

    G4bool fForcedrawphotons;
    G4bool fForcenophotons;

};

#endif
