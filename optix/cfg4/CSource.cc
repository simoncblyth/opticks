#include "CSource.hh"
#include <cstring>

#include "G4Geantino.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"

#include "NLog.hpp"


void CSource::init()
{
	m_definition = G4Geantino::GeantinoDefinition();
}

void CSource::setParticleDefinition(const char* name)
{ 
    m_name = strdup(name);
	m_definition = G4ParticleTable::GetParticleTable()->FindParticle(name);
	m_charge = m_definition->GetPDGCharge();
    m_mass   = m_definition->GetPDGMass();

    LOG(info) << "CSource::setParticleDefinition"
              << " name " << m_name
              << " charge " << m_charge 
              << " mass " << m_mass
              ;

}

