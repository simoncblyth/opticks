#include "CFG4_BODY.hh"
#include <cassert>

// g4-
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4ParticleMomentum.hh"

// cfg4-
#include "CPrimarySource.hh"

#include "PLOG.hh"


CPrimarySource::CPrimarySource(Opticks* ok, int verbosity)  
    :
    CSource(ok, verbosity)
{
    init();
}

void CPrimarySource::init()
{
}

CPrimarySource::~CPrimarySource() 
{
}

void CPrimarySource::GeneratePrimaryVertex(G4Event *evt) 
{
    LOG(fatal) << "CPrimarySource::GeneratePrimaryVertex" ;


/*
    G4ThreeVector position = GetParticlePosition();
    G4double time = GetParticleTime() ; 

    G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);

    G4double energy = GetParticleEnergy();

    G4ParticleMomentum direction = GetParticleMomentumDirection();
    G4ThreeVector polarization = GetParticlePolarization();

    G4double mass = m_definition->GetPDGMass() ;
    G4double charge = m_definition->GetPDGCharge() ;

    G4PrimaryParticle* primary = new G4PrimaryParticle(m_definition);

    primary->SetKineticEnergy( energy);
    primary->SetMass( mass );
    primary->SetMomentumDirection( direction);
    primary->SetCharge( charge );
	primary->SetPolarization(polarization.x(), polarization.y(), polarization.z()); 

    vertex->SetPrimary(primary);
    evt->AddPrimaryVertex(vertex);

 
    LOG(info) << "CPrimarySource::GeneratePrimaryVertex" 
              << " time " << time 
              << " position.x " << position.x() 
              << " position.y " << position.y() 
              << " position.z " << position.z() 
              ; 

*/


}







