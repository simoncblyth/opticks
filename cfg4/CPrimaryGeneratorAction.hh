#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4VPrimaryGenerator ;
class G4Event;
class CSource ; 


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CPrimaryGeneratorAction
=========================

Main method *GeneratePrimaries* invoked by Geant4 beamOn within CG4::propagate
invokes CSource::GeneratePrimaryVertex(G4Event*)

**/


class CPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    CPrimaryGeneratorAction(CSource* generator);
    virtual ~CPrimaryGeneratorAction();
  public:
    virtual void GeneratePrimaries(G4Event*);
  private:
    CSource*  m_source ;

};
#include "CFG4_TAIL.hh"

