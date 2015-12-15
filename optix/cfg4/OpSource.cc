// based on /usr/local/env/g4/geant4.10.02/source/event/include/G4SingleParticleSource.hh 
#include <cmath>

#include "OpSource.hh"

#include "G4SystemOfUnits.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "Randomize.hh"
#include "G4ParticleTable.hh"
#include "G4Geantino.hh"
#include "G4ParticleDefinition.hh"

#include "G4TrackingManager.hh"
#include "G4Track.hh"
#include "G4AutoLock.hh"


#include "G4SPSPosDistribution.hh"
#include "G4SPSAngDistribution.hh"
#include "G4SPSEneDistribution.hh"
#include "G4SPSRandomGenerator.hh"


OpSource::part_prop_t::part_prop_t() 
{
  momentum_direction = G4ParticleMomentum(1,0,0);
  energy = 1.*MeV;
  position = G4ThreeVector();
}

void OpSource::init()
{
	m_definition = G4Geantino::GeantinoDefinition();
	m_biasRndm = new G4SPSRandomGenerator();
	m_posGenerator = new G4SPSPosDistribution();
	m_posGenerator->SetBiasRndm(m_biasRndm);
	m_angGenerator = new G4SPSAngDistribution();
	m_angGenerator->SetPosDistribution(m_posGenerator);
	m_angGenerator->SetBiasRndm(m_biasRndm);
	m_eneGenerator = new G4SPSEneDistribution();
	m_eneGenerator->SetBiasRndm(m_biasRndm);
    
    G4MUTEXINIT(m_mutex);
}


OpSource::~OpSource() 
{
	delete m_biasRndm;
	delete m_posGenerator;
	delete m_angGenerator;
	delete m_eneGenerator;

    G4MUTEXDESTROY(m_mutex);
}

void OpSource::SetVerbosity(int vL) 
{
    G4AutoLock l(&m_mutex);
	m_verbosityLevel = vL;
	m_posGenerator->SetVerbosity(vL);
	m_angGenerator->SetVerbosity(vL);
	m_eneGenerator->SetVerbosity(vL);
}

void OpSource::SetParticleDefinition(G4ParticleDefinition* definition) 
{
	m_definition = definition;
	m_charge = definition->GetPDGCharge();
}

void OpSource::GeneratePrimaryVertex(G4Event *evt) 
{
    assert(m_definition);

	if (m_verbosityLevel > 1)
		G4cout << " NumberOfParticlesToBeGenerated: "
				<< m_num << G4endl;

    part_prop_t& pp = m_pp.Get();

	for (G4int i = 0; i < m_num; i++) 
    {
	    pp.position = m_posGenerator->GenerateOne();
        G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position,m_time);

		pp.momentum_direction = m_angGenerator->GenerateOne();
		pp.energy = m_eneGenerator->GenerateOne(m_definition);

		if (m_verbosityLevel >= 2)
			G4cout << "Creating primaries and assigning to vertex" << G4endl;
		// create new primaries and set them to the vertex
		G4double mass = m_definition->GetPDGMass();

		G4PrimaryParticle* particle = new G4PrimaryParticle(m_definition);
		particle->SetKineticEnergy(pp.energy );
		particle->SetMass( mass );
		particle->SetMomentumDirection( pp.momentum_direction );
		particle->SetCharge( m_charge );

        if(m_isspol)
        {
            G4double phi = std::atan2( pp.position.y(), pp.position.x() ); 
		    particle->SetPolarization(std::sin(phi), -std::cos(phi), 0. );
        }
        else
        {
		    particle->SetPolarization(m_polarization.x(), m_polarization.y(), m_polarization.z());
        }

		if (m_verbosityLevel > 1) {
			G4cout << "Particle name: "
					<< m_definition->GetParticleName() << G4endl;
			G4cout << "       Energy: " << pp.energy << G4endl;
			G4cout << "     Position: " << pp.position << G4endl;
			G4cout << "    Direction: " << pp.momentum_direction
					<< G4endl;
		}
		// Set bweight equal to the multiple of all non-zero weights
		G4double weight = m_eneGenerator->GetWeight()*m_biasRndm->GetBiasWeight();
		// pass it to primary particle
		particle->SetWeight(weight);

		vertex->SetPrimary(particle);

        evt->AddPrimaryVertex(vertex);
        if (m_verbosityLevel > 1)
            G4cout << " Primary Vetex generated !" << G4endl;
	}
}


