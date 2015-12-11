#include "PrimaryGeneratorAction.hh"

#include "Randomize.hh"

#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"


PrimaryGeneratorAction::PrimaryGeneratorAction()
 : 
   G4VUserPrimaryGeneratorAction(), 
   m_gun(NULL)
{
    G4int n_particle = 1;
    m_gun = new G4ParticleGun(n_particle);

    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* particle = particleTable->FindParticle("e+");

    m_gun->SetParticleDefinition(particle);
    m_gun->SetParticleTime(0.0*ns);
    m_gun->SetParticlePosition(G4ThreeVector(0.0*cm,0.0*cm,0.0*cm));
    m_gun->SetParticleMomentumDirection(G4ThreeVector(1.,0.,0.));
    m_gun->SetParticleEnergy(500.0*keV);
}


PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
   delete m_gun;
}


void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
    m_gun->GeneratePrimaryVertex(anEvent);
}


void PrimaryGeneratorAction::SetOpticalPhotonPolarization()
{
    G4double angle = G4UniformRand() * 360.0*deg;
    SetOpticalPhotonPolarization(angle);
}

void PrimaryGeneratorAction::SetOpticalPhotonPolarization(G4double angle)
{

    if (m_gun->GetParticleDefinition()->GetParticleName()!="opticalphoton")
    {
        G4cout << "--> warning from PrimaryGeneratorAction::SetOptPhotonPolar() :"
                   "the particleGun is not an opticalphoton" << G4endl;
        return;
    }

    G4ThreeVector normal (1., 0., 0.);
    G4ThreeVector kphoton = m_gun->GetParticleMomentumDirection();
    G4ThreeVector product = normal.cross(kphoton);
    G4double modul2       = product*product;
     
    G4ThreeVector e_perpend (0., 0., 1.);
    if (modul2 > 0.) e_perpend = (1./std::sqrt(modul2))*product;
    G4ThreeVector e_paralle    = e_perpend.cross(kphoton);
     
    G4ThreeVector polar = std::cos(angle)*e_paralle + std::sin(angle)*e_perpend;

    m_gun->SetParticlePolarization(polar);
}

