#include "PrimaryGeneratorAction.hh"

#include "Format.hh"
#include "RecorderBase.hh"
#include "OpSource.hh"

#include "G4Event.hh"
#include "Randomize.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

void PrimaryGeneratorAction::init()
{
    unsigned int npho = m_recorder->getPhotonsPerEvent();
    m_generator = MakeGenerator(npho);
}
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    m_generator->GeneratePrimaryVertex(event);
}
G4VPrimaryGenerator* PrimaryGeneratorAction::MakeGenerator(unsigned int n)
{
    OpSource* src = new OpSource();
    G4SPSPosDistribution* posGen = src->GetPosDist();
    G4SPSAngDistribution* angGen = src->GetAngDist();
    G4SPSEneDistribution* eneGen = src->GetEneDist();

    G4double wavelength = 550*nm ; 
    G4double energy = h_Planck*c_light/wavelength ;

    G4double time = 0.0*ns ;
    G4ThreeVector pos(-600.0*mm,0.0*mm,0.0*mm);
    G4ThreeVector cen(pos);
    G4ThreeVector dir(1.,0.,0.);
    G4ThreeVector pol(1.,0.,0.);
    G4ThreeVector posX(0,1.,0.);
    G4ThreeVector posY(0,0.,1.);
    G4ParticleDefinition* definition = G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton");

    src->SetNumberOfParticles(n);
    src->SetParticleDefinition(definition);
    src->SetParticleTime(time);
    src->SetParticlePosition(pos);
    src->SetParticlePolarization(pol);

    posGen->SetPosDisType("Plane");
    posGen->SetPosDisShape("Circle");
    posGen->SetRadius(100.0*mm);
    posGen->SetCentreCoords(cen);
    posGen->SetPosRot1(posX);
    posGen->SetPosRot2(posY);
    //for(unsigned int i=0 ; i < 10 ; i++) G4cout << Format(posGen->GenerateOne(), "posGen", 10) << G4endl ; 

    angGen->SetAngDistType("planar");
    angGen->SetParticleMomentumDirection(dir);
    //for(unsigned int i=0 ; i < 10 ; i++) G4cout << Format(angGen->GenerateOne(), "angGen") << G4endl ; 

    eneGen->SetEnergyDisType("Mono");
    eneGen->SetMonoEnergy(energy);

    return src ; 
}


PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete m_generator;
}



