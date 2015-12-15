#pragma once

#include "G4Run.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4OpBoundaryProcess.hh"

#include <glm/glm.hpp>

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
    virtual unsigned int defineRecordId() = 0 ; 

    virtual void setEventId(unsigned int) = 0 ; 
    virtual void setPhotonId(unsigned int) = 0 ; 
    virtual void setRecordId(unsigned int) = 0 ; 
    virtual void setStepId(unsigned int) = 0 ; 
    virtual void startPhoton() = 0 ; 

    virtual void setIncidentSphereSPolarized(bool) = 0 ;
    virtual bool getIncidentSphereSPolarized() = 0 ;

    virtual unsigned int getStepId() = 0 ; 
    virtual void setBoundaryStatus(G4OpBoundaryProcessStatus) = 0;
    virtual void setCenterExtent(const glm::vec4&) = 0;
    virtual void setBoundaryDomain(const glm::vec4&) = 0;
 
    virtual void save() {};

};

