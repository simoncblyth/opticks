#include "CFG4_BODY.hh"
#include <cassert>

// npy-
#include "NGunConfig.hpp"

// g4-
#include "G4AutoLock.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4ParticleMomentum.hh"

// cfg4-
#include "CRecorder.hh"
#include "CGunSource.hh"

#include "PLOG.hh"



CGunSource::CGunSource(Opticks* ok, int verbosity)  
    :
    CSource(ok, verbosity),
    m_config(NULL)
{
    init();
}


void CGunSource::init()
{

  //  G4MUTEXINIT(m_mutex);
}

CGunSource::~CGunSource() 
{
  //  G4MUTEXDESTROY(m_mutex);
}

void CGunSource::SetVerbosity(int vL) 
{
  //  G4AutoLock l(&m_mutex);
    m_verbosityLevel = vL;
}

void CGunSource::configure(NGunConfig* gc)
{
    m_config = gc ; 

    LOG(info) << "CGunSource::configure" ; 
    gc->Summary("CGunSource::configure");

    setParticle(gc->getParticle());
    assert(m_definition);

    SetParticleTime( gc->getTime()*ns );
    SetParticleEnergy( gc->getEnergy()*MeV );

    glm::vec3 pos = gc->getPosition();
    glm::vec3 dir = gc->getDirection();
    glm::vec3 pol = gc->getPolarization();

    SetParticlePosition(G4ThreeVector(pos.x*mm,pos.y*mm,pos.z*mm));
    SetParticleMomentumDirection(G4ThreeVector(dir.x,dir.y,dir.z));
    SetParticlePolarization(G4ThreeVector(pol.x,pol.y,pol.z));

}

void CGunSource::GeneratePrimaryVertex(G4Event *evt) 
{
    LOG(fatal) << "CGunSource::GeneratePrimaryVertex" ;

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

 
    LOG(info) << "CGunSource::GeneratePrimaryVertex" 
              << " time " << time 
              << " position.x " << position.x() 
              << " position.y " << position.y() 
              << " position.z " << position.z() 
              ; 

    //m_recorder->RecordPrimaryVertex(vertex);

}

