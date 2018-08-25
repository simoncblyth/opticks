#pragma once

class Opticks ; 
class CRecorder ; 

class G4Event ; 
class G4PrimaryVertex ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CSource(G4VPrimaryGenerator) : common functionality of the various source types
=======================================================================================

* abstract base class of CTorchSource, CGunSource, CInputPhotonSource 
* subclass of G4VPrimaryGenerator

The specialized prime method GeneratePrimaryVertex 
is invoked from CPrimaryGeneratorAction::GeneratePrimaries
by the Geant4 framework.

**/

#include "G4VPrimaryGenerator.hh"

class CFG4_API CSource : public G4VPrimaryGenerator
{
  public:
    friend class CTorchSource ; 
    friend class CGunSource ; 
  public:
    CSource(Opticks* ok );
    void setRecorder(CRecorder* recorder);
    virtual ~CSource();
  public:
    virtual void GeneratePrimaryVertex(G4Event *evt) = 0 ;
  public:
     // to CPrimaryCollector
    void collectPrimaryVertex(const G4PrimaryVertex* vtx);
  protected: 
    Opticks*              m_ok ;  
    CRecorder*            m_recorder ; 
};
#include "CFG4_TAIL.hh"

