#include "CSource.hh"
#include <cstring>

#include "G4Geantino.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"


#include "PLOG.hh"



CSource::part_prop_t::part_prop_t() 
{
  momentum_direction = G4ParticleMomentum(0,0,-1);
  energy = 1.*MeV;
  position = G4ThreeVector();
}



void CSource::init()
{
	m_definition = G4Geantino::GeantinoDefinition();
}

void CSource::SetVerbosity(int vL) 
{
    G4AutoLock l(&m_mutex);
	m_verbosityLevel = vL;
}

void CSource::setParticle(const char* name)
{ 
	G4ParticleDefinition* definition = G4ParticleTable::GetParticleTable()->FindParticle(name);
    SetParticleDefinition(definition);
}

void CSource::SetParticleDefinition(G4ParticleDefinition* definition)
{ 
    m_definition = definition ; 
}





