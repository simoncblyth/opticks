#pragma once

#include "G4ParticleMomentum.hh"

#include "G4Threading.hh"
#include "G4Cache.hh"

class G4ParticleDefinition ; 
class G4SPSPosDistribution ;
class G4SPSAngDistribution ;
class G4SPSEneDistribution ;
class G4SPSRandomGenerator ;

class TorchStepNPY ; 
class Recorder ; 

#include "CSource.hh"

class CTorchSource: public CSource
{
  public:
    CTorchSource(TorchStepNPY* torch, unsigned int verbosity);
  private:
    void init();
    void configure();
  public:
    virtual ~CTorchSource();
    void GeneratePrimaryVertex(G4Event *evt);
  public:
    void SetVerbosity(int vL);
    void SetParticleDefinition(G4ParticleDefinition* definition);
    void SetNumberOfParticles(G4int i);
    void SetParticleTime(G4double time);
    void SetParticleCharge(G4double charge);
    void SetParticlePolarization(G4ThreeVector polarization);
  public:
    G4int                 GetNumberOfParticles() const ;
    G4ParticleDefinition* GetParticleDefinition() const ;
    G4ThreeVector         GetParticlePolarization() const ;
    G4ThreeVector         GetParticlePosition() const ;
    G4ThreeVector         GetParticleMomentumDirection() const ;
    G4double              GetParticleEnergy() const ;
    G4double              GetParticleTime() const;
private:
    // residents 
    TorchStepNPY*         m_torch ;

    G4SPSPosDistribution* m_posGen;
    G4SPSAngDistribution* m_angGen;
    G4SPSEneDistribution* m_eneGen;
    G4SPSRandomGenerator* m_ranGen;

    struct part_prop_t 
    {
        G4ParticleMomentum momentum_direction; 
        G4double           energy; 
        G4ThreeVector      position; 
        part_prop_t();
    };

    G4Cache<part_prop_t>  m_pp;

    G4int                 m_num;
    G4double              m_time;
    G4ThreeVector         m_polarization;
    G4int                 m_verbosityLevel;
    G4Mutex               m_mutex;
};


inline CTorchSource::CTorchSource(TorchStepNPY* torch, unsigned int verbosity)  
    :
    CSource(),
    m_torch(torch),
    m_posGen(NULL),
    m_angGen(NULL),
    m_eneGen(NULL),
    m_ranGen(NULL),
	m_num(1),

	m_time(0.0),
	m_polarization(1.0,0.0,0.0),
    m_verbosityLevel(verbosity)
{
    init();
}



inline void CTorchSource::SetNumberOfParticles(G4int num) 
{
    m_num = num;
}
inline void CTorchSource::SetParticleTime(G4double time) 
{
    m_time = time;
}
inline void CTorchSource::SetParticleCharge(G4double charge) 
{
    m_charge = charge;
}
inline void CTorchSource::SetParticlePolarization(G4ThreeVector polarization) 
{
    m_polarization = polarization ;
}




inline G4ParticleDefinition* CTorchSource::GetParticleDefinition() const 
{
    return m_definition;
}
inline G4int CTorchSource::GetNumberOfParticles() const 
{
    return m_num ;
}
inline G4double CTorchSource::GetParticleTime() const 
{
    return m_time;
}
inline G4ThreeVector CTorchSource::GetParticlePolarization() const 
{
    return m_polarization;
}



inline G4ThreeVector CTorchSource::GetParticlePosition() const 
{
    return m_pp.Get().position;
}
inline G4ThreeVector CTorchSource::GetParticleMomentumDirection() const 
{
    return m_pp.Get().momentum_direction;
}
inline G4double CTorchSource::GetParticleEnergy() const 
{
    return m_pp.Get().energy;
}




