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
#include "G4IonTable.hh"
#include "G4Ions.hh"
#include "G4TrackingManager.hh"
#include "G4Track.hh"
#include "G4AutoLock.hh"

OpSource::part_prop_t::part_prop_t() {
  //definition = G4Geantino::GeantinoDefinition();
  momentum_direction = G4ParticleMomentum(1,0,0);
  energy = 1.*MeV;
  position = G4ThreeVector();
}

OpSource::OpSource() {
	NumberOfParticlesToBeGenerated = 1;
	definition = G4Geantino::GeantinoDefinition();

	charge = 0.0;
	time = 0;
	polarization = G4ThreeVector();

	biasRndm = new G4SPSRandomGenerator();
	posGenerator = new G4SPSPosDistribution();
	posGenerator->SetBiasRndm(biasRndm);
	angGenerator = new G4SPSAngDistribution();
	angGenerator->SetPosDistribution(posGenerator);
	angGenerator->SetBiasRndm(biasRndm);
	eneGenerator = new G4SPSEneDistribution();
	eneGenerator->SetBiasRndm(biasRndm);

	verbosityLevel = 0;
    
    G4MUTEXINIT(mutex);

}

OpSource::~OpSource() {
	delete biasRndm;
	delete posGenerator;
	delete angGenerator;
	delete eneGenerator;
    G4MUTEXDESTROY(mutex);
}

void OpSource::SetVerbosity(int vL) {
        G4AutoLock l(&mutex);
	verbosityLevel = vL;
	posGenerator->SetVerbosity(vL);
	angGenerator->SetVerbosity(vL);
	eneGenerator->SetVerbosity(vL);
	//G4cout << "Verbosity Set to: " << verbosityLevel << G4endl;
}

void OpSource::SetParticleDefinition(
		G4ParticleDefinition* aParticleDefinition) {
	definition = aParticleDefinition;
	charge = aParticleDefinition->GetPDGCharge();
}

void OpSource::GeneratePrimaryVertex(G4Event *evt) 
{
    assert(definition);

	if (verbosityLevel > 1)
		G4cout << " NumberOfParticlesToBeGenerated: "
				<<NumberOfParticlesToBeGenerated << G4endl;

    part_prop_t& pp = ParticleProperties.Get();

	for (G4int i = 0; i < NumberOfParticlesToBeGenerated; i++) 
    {
	    pp.position = posGenerator->GenerateOne();
        G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position,time);

		pp.momentum_direction = angGenerator->GenerateOne();
		pp.energy = eneGenerator->GenerateOne(definition);

		if (verbosityLevel >= 2)
			G4cout << "Creating primaries and assigning to vertex" << G4endl;
		// create new primaries and set them to the vertex
		G4double mass = definition->GetPDGMass();
		G4PrimaryParticle* particle =
		  new G4PrimaryParticle(definition);
		particle->SetKineticEnergy(pp.energy );
		particle->SetMass( mass );
		particle->SetMomentumDirection( pp.momentum_direction );
		particle->SetCharge( charge );
		particle->SetPolarization(polarization.x(),
					  polarization.y(),
					  polarization.z());
		if (verbosityLevel > 1) {
			G4cout << "Particle name: "
					<< definition->GetParticleName() << G4endl;
			G4cout << "       Energy: " << pp.energy << G4endl;
			G4cout << "     Position: " << pp.position << G4endl;
			G4cout << "    Direction: " << pp.momentum_direction
					<< G4endl;
		}
		// Set bweight equal to the multiple of all non-zero weights
		G4double weight = eneGenerator->GetWeight()*biasRndm->GetBiasWeight();
		// pass it to primary particle
		particle->SetWeight(weight);

		vertex->SetPrimary(particle);

        evt->AddPrimaryVertex(vertex);
        if (verbosityLevel > 1)
            G4cout << " Primary Vetex generated !" << G4endl;
	}
}



