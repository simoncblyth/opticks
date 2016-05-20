#pragma once

class Recorder ; 
class G4Event ; 
class G4ParticleDefinition ;

#include "G4VPrimaryGenerator.hh"
class CSource : public G4VPrimaryGenerator
{
  public:
    friend class CTorchSource ; 
    friend class CGunSource ; 
  public:
    CSource();
    void setRecorder(Recorder* recorder);
    void setParticleDefinition(const char* name);
    virtual ~CSource();
  public:
    virtual void GeneratePrimaryVertex(G4Event *evt) = 0 ;
  private:
    void init();
  private:
    Recorder*             m_recorder ; 
    const char*           m_name ; 
    G4ParticleDefinition* m_definition;
    G4double              m_charge;
    G4double              m_mass ;
 
};


inline CSource::CSource()  
    :
    m_recorder(NULL),
    m_name(NULL),
	m_charge(0.0),
	m_mass(0.0)
{
    init();
}

inline void CSource::setRecorder(Recorder* recorder)
{
   m_recorder = recorder ;  
}


inline CSource::~CSource()
{
}  

