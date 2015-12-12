#pragma once

#include "G4Run.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4Step.hh"

class RecorderBase {
  public:
    virtual ~RecorderBase() {};

    virtual void RecordBeginOfRun(const G4Run*) = 0;
    virtual void RecordEndOfRun(const G4Run*) = 0;
    virtual void RecordBeginOfEvent(const G4Event*) {};
    virtual void RecordEndOfEvent(const G4Event*) {};
    virtual void RecordTrack(const G4Track*) {};
    virtual void RecordStep(const G4Step*) {};

    virtual void save(const char*) {};

};

