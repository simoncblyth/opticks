#pragma once

class Opticks ; 

class CRecorder ; 
class G4Event ; 
class G4ParticleDefinition ;
class G4PrimaryVertex ; 

#include "G4ParticleMomentum.hh"
#include "G4Threading.hh"
#include "G4Cache.hh"
#include "G4VPrimaryGenerator.hh"

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


class CFG4_API CSource : public G4VPrimaryGenerator
{
  public:
    friend class CTorchSource ; 
    friend class CGunSource ; 
  public:
    CSource(Opticks* ok, int verbosity);
    void setRecorder(CRecorder* recorder);
    void setParticle(const char* name);
    virtual ~CSource();
  public:
    virtual void GeneratePrimaryVertex(G4Event *evt) = 0 ;
  private:
    void init();

    virtual void SetVerbosity(int vL);
  public:
     // inline setters
    void SetNumberOfParticles(G4int i);
    void SetParticleDefinition(G4ParticleDefinition* definition);
    void SetParticlePolarization(G4ThreeVector polarization);
    void SetParticlePosition(G4ThreeVector position);
    void SetParticleMomentumDirection(G4ThreeVector direction);
    void SetParticleEnergy(G4double energy);
    void SetParticleTime(G4double time);
 public:
    void collectPrimary(G4PrimaryVertex* vertex);
 public:
    // inline getters
    G4int                 GetNumberOfParticles() const ;
    G4ParticleDefinition* GetParticleDefinition() const ;
    G4ThreeVector         GetParticlePolarization() const ;
    G4ThreeVector         GetParticlePosition() const ;
    G4ThreeVector         GetParticleMomentumDirection() const ;
    G4double              GetParticleEnergy() const ;
    G4double              GetParticleTime() const;
  protected: 
    Opticks*              m_ok ;  
    CRecorder*            m_recorder ; 
    G4int                 m_num;
    G4ParticleDefinition* m_definition;

    struct part_prop_t 
    {
        G4ParticleMomentum momentum_direction; 
        G4double           energy; 
        G4ThreeVector      position; 
        part_prop_t();
    };
    G4Cache<part_prop_t>  m_pp;

    G4double              m_time;
    G4ThreeVector         m_polarization;
    G4int                 m_verbosityLevel;
    G4Mutex               m_mutex;
};
#include "CFG4_TAIL.hh"

