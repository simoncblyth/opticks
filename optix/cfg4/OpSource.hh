#pragma once

#include "G4VPrimaryGenerator.hh"
#include "G4ParticleMomentum.hh"
#include "G4ParticleDefinition.hh"

#include "G4SPSPosDistribution.hh"
#include "G4SPSAngDistribution.hh"
#include "G4SPSEneDistribution.hh"
#include "G4SPSRandomGenerator.hh"
#include "G4Threading.hh"
#include "G4Cache.hh"

class OpSource: public G4VPrimaryGenerator {
public:
	OpSource();
	~OpSource();

	void GeneratePrimaryVertex(G4Event *evt);
	//

	G4SPSPosDistribution* GetPosDist() const {
		return posGenerator;
	}
	;
	G4SPSAngDistribution* GetAngDist() const {
		return angGenerator;
	}
	;
	G4SPSEneDistribution* GetEneDist() const {
		return eneGenerator;
	}
	;
	G4SPSRandomGenerator* GetBiasRndm() const {
		return biasRndm;
	}
	;

	// Set the verbosity level.
	void SetVerbosity(G4int);

	// Set the particle species
	void SetParticleDefinition(G4ParticleDefinition * aParticleDefinition);
	inline G4ParticleDefinition * GetParticleDefinition() const {
		return definition;
	}
	;

	inline void SetParticleCharge(G4double aCharge) {
	        charge = aCharge;
	}
	;

	// Set polarization
	inline void SetParticlePolarization(G4ThreeVector aVal) {
	  polarization = aVal;
	}
	;
	inline G4ThreeVector GetParticlePolarization() const {
		return polarization;
	}
	;

	// Set Time.
	inline void SetParticleTime(G4double aTime) {
	  time = aTime;
	}
	;
	inline G4double GetParticleTime() const {
		return time;
	}
	;

	inline void SetNumberOfParticles(G4int i) {
	  NumberOfParticlesToBeGenerated = i;
	}
	;
	//
	inline G4int GetNumberOfParticles() const {
		return NumberOfParticlesToBeGenerated;
	}
	;
	inline G4ThreeVector GetParticlePosition() const {
		return ParticleProperties.Get().position;
	}
	;
	inline G4ThreeVector GetParticleMomentumDirection() const {
		return ParticleProperties.Get().momentum_direction;
	}
	;
	inline G4double GetParticleEnergy() const {
		return ParticleProperties.Get().energy;
	}
	;

private:

	G4SPSPosDistribution* posGenerator;
	G4SPSAngDistribution* angGenerator;
	G4SPSEneDistribution* eneGenerator;
	G4SPSRandomGenerator* biasRndm;

	struct part_prop_t 
    {
	    G4ParticleMomentum momentum_direction; 
	    G4double energy; 
	    G4ThreeVector position; 
	    part_prop_t();
	};

	G4Cache<part_prop_t> ParticleProperties;
    G4int NumberOfParticlesToBeGenerated;
    G4ParticleDefinition * definition;
    G4double charge;
    G4double time;
    G4ThreeVector polarization;

	G4int verbosityLevel;

    G4Mutex mutex;
};


