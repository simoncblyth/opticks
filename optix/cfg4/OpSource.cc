// based on /usr/local/env/g4/geant4.10.02/source/event/include/G4SingleParticleSource.hh 
#include <cmath>

// npy-
#include "TorchStepNPY.hpp"
#include "NLog.hpp"

// cfg4-
#include "OpSource.hh"

// g4-
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

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
  momentum_direction = G4ParticleMomentum(0,0,-1);
  energy = 1.*MeV;
  position = G4ThreeVector();
}

void OpSource::init()
{
	m_definition = G4Geantino::GeantinoDefinition();

	m_ranGen = new G4SPSRandomGenerator();

	m_posGen = new G4SPSPosDistribution();
	m_posGen->SetBiasRndm(m_ranGen);

	m_angGen = new G4SPSAngDistribution();
	m_angGen->SetPosDistribution(m_posGen);
	m_angGen->SetBiasRndm(m_ranGen);

	m_eneGen = new G4SPSEneDistribution();
	m_eneGen->SetBiasRndm(m_ranGen);

    configure();
    
    G4MUTEXINIT(m_mutex);
}


OpSource::~OpSource() 
{
	delete m_ranGen;
	delete m_posGen;
	delete m_angGen;
	delete m_eneGen;

    G4MUTEXDESTROY(m_mutex);
}

void OpSource::SetVerbosity(int vL) 
{
    G4AutoLock l(&m_mutex);
	m_verbosityLevel = vL;
	m_posGen->SetVerbosity(vL);
	m_angGen->SetVerbosity(vL);
	m_eneGen->SetVerbosity(vL);
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

    bool incidentSphere = m_torch->isIncidentSphere() ;
    bool reflTest = m_torch->isReflTest() ;

    bool SPol =  m_torch->isSPolarized() ;
    bool PPol =  m_torch->isPPolarized() ;

    
    LOG(debug) << "OpSource::GeneratePrimaryVertex"
              << " incidentSphere " << incidentSphere
              << " reflTest " << reflTest
              << " SPol " << SPol 
              << " PPol " << PPol 
              << " num " << m_num 
              ;


	for (G4int i = 0; i < m_num; i++) 
    {
	    pp.position = m_posGen->GenerateOne();
        G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position,m_time);

		pp.momentum_direction = m_angGen->GenerateOne();
		pp.energy = m_eneGen->GenerateOne(m_definition);

		if (m_verbosityLevel >= 2)
			G4cout << "Creating primaries and assigning to vertex" << G4endl;
		// create new primaries and set them to the vertex
		G4double mass = m_definition->GetPDGMass();


		G4PrimaryParticle* particle = new G4PrimaryParticle(m_definition);
		particle->SetKineticEnergy(pp.energy );
		particle->SetMass( mass );
		particle->SetMomentumDirection( pp.momentum_direction );
		particle->SetCharge( m_charge );

        if(incidentSphere)
        {
            // custom S/P-polarization specific to "rainbow geometry" 
            // (planar disc of photons incident on sphere with same radius as disc)
            //
            G4ThreeVector tangent(-pp.position.y(),  pp.position.x(), 0. );  // anti-clockwise tangent
            G4ThreeVector radial(  pp.position.x(),  pp.position.y(), 0. ); 
            if(SPol)
		       particle->SetPolarization(tangent.unit()); 
            else if(PPol)
		       particle->SetPolarization(radial.unit());  
            else
		       particle->SetPolarization(m_polarization.x(), m_polarization.y(), m_polarization.z());

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
		G4double weight = m_eneGen->GetWeight()*m_ranGen->GetBiasWeight();
		particle->SetWeight(weight);

		vertex->SetPrimary(particle);

        evt->AddPrimaryVertex(vertex);
	}
}



void OpSource::configure()
{
    m_torch->Summary("OpSource::configure");

    unsigned int n = m_torch->getNumPhotonsPerG4Event();
    SetNumberOfParticles(n);

    G4ParticleDefinition* definition = G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton");
    SetParticleDefinition(definition);



    float w = m_torch->getWavelength() ; 
    if(w > 0.f)
    {
        G4double wavelength = w*nm ; 
        G4double energy = h_Planck*c_light/wavelength ;
        m_eneGen->SetEnergyDisType("Mono");
        m_eneGen->SetMonoEnergy(energy);
    }
    else
    {
        assert(0 && "only mono supported"); 
    }



    float _t = m_torch->getTime();
    glm::vec3 _pos = m_torch->getPosition();
    glm::vec3 _dir = m_torch->getDirection();
    float _radius = m_torch->getRadius();

    SetParticleTime(_t*ns);
    G4ThreeVector pos(_pos.x*mm,_pos.y*mm,_pos.z*mm);
    SetParticlePosition(pos);

    G4ThreeVector cen(pos);

    // TODO: from config? need more state as the pol holds surfaceNormal ?
    G4ThreeVector pol(1.,0.,0.); 
    SetParticlePolarization(pol); // reset later for the custom configs 

    
    bool incidentSphere = m_torch->isIncidentSphere() ;
    bool reflTest = m_torch->isReflTest() ;

    // for sanity test geometries use standard X Y for 

    G4ThreeVector posX(1,0.,0.);
    G4ThreeVector posY(0,1.,0.);
    m_posGen->SetPosRot1(posX);
    m_posGen->SetPosRot2(posY);


    if(incidentSphere)  // "rainbow" geometry 
    {
        m_posGen->SetPosDisType("Plane");
        m_posGen->SetPosDisShape("Circle");
        m_posGen->SetRadius(_radius*mm);
        m_posGen->SetCentreCoords(cen);

        m_angGen->SetAngDistType("planar");
        G4ThreeVector dir(_dir.x,_dir.y,_dir.z);
        m_angGen->SetParticleMomentumDirection(dir);

    } 
    else if( reflTest ) 
    {
        m_posGen->SetPosDisType("Surface");
        m_posGen->SetPosDisShape("Sphere");
        m_posGen->SetRadius(_radius*mm);
        m_posGen->SetCentreCoords(cen);

        m_angGen->SetAngDistType("focused");
        m_angGen->SetFocusPoint(cen); 
    }

    //for(unsigned int i=0 ; i < 10 ; i++) G4cout << Format(posGen->GenerateOne(), "posGen", 10) << G4endl ; 
    //for(unsigned int i=0 ; i < 10 ; i++) G4cout << Format(angGen->GenerateOne(), "angGen") << G4endl ; 

}



