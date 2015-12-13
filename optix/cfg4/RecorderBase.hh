#pragma once

#include "G4Run.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4OpBoundaryProcess.hh"

class RecorderBase {
  public:
    virtual ~RecorderBase() {};

    virtual void RecordBeginOfRun(const G4Run*) = 0;
    virtual void RecordEndOfRun(const G4Run*) = 0;
    virtual void RecordBeginOfEvent(const G4Event*) {};
    virtual void RecordEndOfEvent(const G4Event*) {};
    virtual void RecordTrack(const G4Track*) {};
    virtual void RecordStep(const G4Step*) {};

    virtual unsigned int getPhotonsPerEvent() = 0 ; 
    virtual unsigned int getRecordMax() = 0 ; 
    virtual unsigned int getPhotonId() = 0 ; 
    virtual unsigned int getStepId() = 0 ; 
    virtual void setStepStatus(G4OpBoundaryProcessStatus) = 0;
 
    virtual void save(const char*) {};

};

