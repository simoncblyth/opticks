#include "CFG4_BODY.hh"
#include <cmath>

// npy-
#include "NPY.hpp"
#include "GLMFormat.hpp"
#include "GenstepNPY.hpp"

#include "Opticks.hh"

// cfg4-
#include "CInputPhotonSource.hh"
#include "CRecorder.hh"


// g4-
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"


#include "G4TrackingManager.hh"
#include "G4Track.hh"


#include "PLOG.hh"


CInputPhotonSource::CInputPhotonSource(Opticks* ok, NPY<float>* input_photons, GenstepNPY* gsnpy, unsigned int verbosity)  
    :
    CSource(ok, verbosity),
    m_sourcedbg(ok->isDbgSource()),
    m_input_photons(input_photons),
    m_gsnpy(gsnpy), 
    m_primary(NPY<float>::make(0,4,4))
{
    init();
}


void CInputPhotonSource::init()
{

}


CInputPhotonSource::~CInputPhotonSource() 
{
}



void CInputPhotonSource::configure()
{
    unsigned numPhotons = m_input_photons->getNumItems();
    unsigned numPhotonsPerG4Event = m_gsnpy->getNumPhotonsPerG4Event();
    unsigned n = numPhotons < numPhotonsPerG4Event ? numPhotons : numPhotonsPerG4Event ;

    if(m_sourcedbg)
    {
        LOG(info) << "[--sourcedbg] CInputPhotonSource::configure" 
                  << " gsnpy " << m_gsnpy->brief()
                  << " numPhotons " << numPhotons 
                  << " numPhotonsPerG4Event " << numPhotonsPerG4Event
                  << " n " << n 
                  ;
    }

    SetNumberOfParticles(n);

    setParticle("opticalphoton");

}




void CInputPhotonSource::GeneratePrimaryVertex(G4Event *evt) 
{


/*
    part_prop_t& pp = m_pp.Get();

	for (G4int i = 0; i < m_num; i++) 
    {
	    pp.position = m_posGen->GenerateOne();

        G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position,m_time);

		pp.momentum_direction = m_angGen->GenerateOne();

		pp.energy = m_eneGen->GenerateOne(m_definition);

        if(m_torchdbg && i < 10) 
        LOG(info) << "CInputPhotonSource::GeneratePrimaryVertex"
                  << " i " << std::setw(6) << i 
                  << " posx " << pp.position.x()
                  << " posy " << pp.position.y()
                  << " posz " << pp.position.z()
                  << " dirx " << pp.momentum_direction.x()
                  << " diry " << pp.momentum_direction.y()
                  << " dirz " << pp.momentum_direction.z()
                  ;


		//if (m_verbosityLevel >= 2)
		//	G4cout << "Creating primaries and assigning to vertex" << G4endl;
		// create new primaries and set them to the vertex


        G4double mass = m_definition->GetPDGMass();
        G4double charge = m_definition->GetPDGCharge();

		G4PrimaryParticle* particle = new G4PrimaryParticle(m_definition);
		particle->SetKineticEnergy(pp.energy );
		particle->SetMass( mass );
		particle->SetMomentumDirection( pp.momentum_direction );
		particle->SetCharge( charge );

	    particle->SetWeight(weight);

		vertex->SetPrimary(particle);
        evt->AddPrimaryVertex(vertex);

        collectPrimary(vertex);

	}
*/


}


