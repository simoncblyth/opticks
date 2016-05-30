#pragma once

class CRecorder ; 
class G4Event ; 
class G4ParticleDefinition ;

#include "G4ParticleMomentum.hh"
#include "G4Threading.hh"
#include "G4Cache.hh"
#include "G4VPrimaryGenerator.hh"

class CSource : public G4VPrimaryGenerator
{
  public:
    friend class CTorchSource ; 
    friend class CGunSource ; 
  public:
    CSource(int verbosity);
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
    // inline getters
    G4int                 GetNumberOfParticles() const ;
    G4ParticleDefinition* GetParticleDefinition() const ;
    G4ThreeVector         GetParticlePolarization() const ;
    G4ThreeVector         GetParticlePosition() const ;
    G4ThreeVector         GetParticleMomentumDirection() const ;
    G4double              GetParticleEnergy() const ;
    G4double              GetParticleTime() const;
  private:
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


inline CSource::CSource(int verbosity)  
    :
    m_recorder(NULL),
	m_num(1),
	m_time(0.0),
	m_polarization(1.0,0.0,0.0),
    m_verbosityLevel(verbosity)
{
    init();
}


inline CSource::~CSource()
{
}  

inline void CSource::setRecorder(CRecorder* recorder)
{
   m_recorder = recorder ;  
}




inline void CSource::SetNumberOfParticles(G4int num) 
{
    m_num = num;
}
inline void CSource::SetParticleTime(G4double time) 
{
    m_time = time;
}
inline void CSource::SetParticlePolarization(G4ThreeVector polarization) 
{
    m_polarization = polarization ;
}
inline void CSource::SetParticlePosition(G4ThreeVector position) 
{
    part_prop_t& pp = m_pp.Get();
    pp.position = position ; 
}
inline void CSource::SetParticleMomentumDirection(G4ThreeVector direction) 
{
    part_prop_t& pp = m_pp.Get();
    pp.momentum_direction = direction  ; 
}
inline void CSource::SetParticleEnergy(G4double energy) 
{
    part_prop_t& pp = m_pp.Get();
    pp.energy = energy ; 
}



inline G4int CSource::GetNumberOfParticles() const 
{
    return m_num ;
}
inline G4ParticleDefinition* CSource::GetParticleDefinition() const 
{
    return m_definition;
}
inline G4double CSource::GetParticleTime() const 
{
    return m_time;
}

inline G4ThreeVector CSource::GetParticlePolarization() const 
{
    return m_polarization;
}
inline G4ThreeVector CSource::GetParticlePosition() const 
{
    return m_pp.Get().position;
}
inline G4ThreeVector CSource::GetParticleMomentumDirection() const 
{
    return m_pp.Get().momentum_direction;
}
inline G4double CSource::GetParticleEnergy() const 
{
    return m_pp.Get().energy;
}


