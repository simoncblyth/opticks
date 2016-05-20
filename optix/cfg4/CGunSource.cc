#include "CGunSource.hh"

#include <cassert>

// cfg4-
#include "Recorder.hh"

// g4-
#include "G4AutoLock.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4ParticleMomentum.hh"


void CGunSource::init()
{
    configure();
    G4MUTEXINIT(m_mutex);
}

CGunSource::~CGunSource() 
{
    G4MUTEXDESTROY(m_mutex);
}

void CGunSource::SetVerbosity(int vL) 
{
    G4AutoLock l(&m_mutex);
    m_verbosityLevel = vL;
}

void CGunSource::configure()
{
    setParticleDefinition("e+");
}

void CGunSource::GeneratePrimaryVertex(G4Event *evt) 
{
    assert(m_definition);

    G4double time = 0.1*ns ; 
    G4ThreeVector position(0,0,0);
    G4double energy = 2.*MeV;

    G4ParticleMomentum direction(0,0,1);

    G4PrimaryParticle* particle = new G4PrimaryParticle(m_definition);
    particle->SetMass( m_mass );
    particle->SetCharge( m_charge );

    particle->SetKineticEnergy(energy);
    particle->SetMomentumDirection(direction);

    G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);
    vertex->SetPrimary(particle);

    m_recorder->RecordPrimaryVertex(vertex);

    evt->AddPrimaryVertex(vertex);
}

