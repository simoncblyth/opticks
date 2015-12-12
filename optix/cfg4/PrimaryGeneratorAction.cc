#include "PrimaryGeneratorAction.hh"

#include "Format.hh"
#include "G4Event.hh"

#include "G4ParticleGun.hh"
#include "G4SingleParticleSource.hh"
#include "Randomize.hh"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"


PrimaryGeneratorAction::PrimaryGeneratorAction()
 : 
   G4VUserPrimaryGeneratorAction(), 
   m_generator(NULL)
{
   m_generator = MakeGenerator(10);
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    m_generator->GeneratePrimaryVertex(event);
}

G4VPrimaryGenerator* PrimaryGeneratorAction::MakeGenerator(G4int n)
{
    G4double wavelength = 550*nm ; 
    G4double energy = h_Planck*c_light/wavelength ;
    G4double time = 0.0*ns ;
    G4ThreeVector pos(-600.0*mm,0.0*mm,0.0*mm);
    G4ThreeVector dir(1.,0.,0.);
    G4ThreeVector pol(1.,0.,0.);

    G4ThreeVector posX(0,1.,0.);
    G4ThreeVector posY(0,0.,1.);

    G4ParticleDefinition* definition = G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton");

    G4cout << "PrimaryGeneratorAction::MakeGenerator" 
           << " wavelength " << wavelength/nm << " nm "   
           << " energy " << energy/eV << " eV " 
           << G4endl ; 

    /*
    G4ParticleGun* gun = new G4ParticleGun(n);
    gun->SetParticleDefinition(definition);
    gun->SetParticleTime(time);
    gun->SetParticlePosition(pos);
    gun->SetParticleMomentumDirection(dir);
    gun->SetParticlePolarization(pol);
    gun->SetParticleEnergy(energy);
    */ 

    G4SingleParticleSource* sps = new G4SingleParticleSource();
    sps->SetNumberOfParticles(n);
    sps->SetParticleDefinition(definition);
    sps->SetParticleTime(time);
    sps->SetParticlePosition(pos);
    sps->SetParticlePolarization(pol);


    G4SPSPosDistribution* posDis = sps->GetPosDist();
    posDis->SetPosDisType("Plane");
    posDis->SetPosDisShape("Circle");
    posDis->SetRadius(100.0*mm);
    posDis->SetCentreCoords(pos);
    posDis->SetPosRot1(posX);
    posDis->SetPosRot2(posY);

    for(unsigned int i=0 ; i < 10 ; i++)
    {
        G4ThreeVector posSample = posDis->GenerateOne();
        G4cout << Format(posSample, "posSample") << G4endl ; 
    }


    G4SPSAngDistribution* angDis = sps->GetAngDist();
    angDis->SetAngDistType("planar");
    angDis->SetParticleMomentumDirection(dir);

    for(unsigned int i=0 ; i < 10 ; i++)
    {
        G4ParticleMomentum momSample = angDis->GenerateOne();
        G4cout << Format(momSample, "momSample") << G4endl ; 
    }


    G4SPSEneDistribution* eneDis = sps->GetEneDist();
    eneDis->SetEnergyDisType("Mono");
    eneDis->SetMonoEnergy(energy);

    //sps->SetParticleMomentumDirection(dir);
    //sps->SetParticleEnergy(energy);

    return sps ;
}


PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
   delete m_generator;
}



