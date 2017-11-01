#include "CFG4_BODY.hh"
#include <cmath>

// sysrap-
#include "STranche.hh"

// npy-
#include "NPY.hpp"
#include "NPho.hpp"
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

unsigned CInputPhotonSource::getNumG4Event() const 
{
    return m_tranche->num_tranche ; 
}
unsigned CInputPhotonSource::getNumPhotonsPerG4Event() const
{
    return m_numPhotonsPerG4Event ;
}


CInputPhotonSource::CInputPhotonSource(Opticks* ok, NPY<float>* input_photons, GenstepNPY* gsnpy, unsigned int verbosity)  
    :
    CSource(ok, verbosity),
    m_sourcedbg(ok->isDbgSource()),
    m_pho(new NPho(input_photons)),
    m_gsnpy(gsnpy), 
    m_numPhotonsPerG4Event(m_gsnpy->getNumPhotonsPerG4Event()),
    m_numPhotons(m_pho->getNumPhotons()),
    m_tranche(new STranche(m_numPhotons,m_numPhotonsPerG4Event)),
    m_primary(NPY<float>::make(0,4,4)),
    m_gpv_count(0)
{
    setParticle("opticalphoton");
}


CInputPhotonSource::~CInputPhotonSource() 
{
}



G4PrimaryVertex* CInputPhotonSource::convertPhoton(unsigned pho_index)
{
    part_prop_t& pp = m_pp.Get();

    glm::vec4 post = m_pho->getPositionTime(pho_index) ; 
    glm::vec4 dirw = m_pho->getDirectionWeight(pho_index) ; 
    glm::vec4 polw = m_pho->getPolarizationWavelength(pho_index) ; 

    pp.position.set(post.x, post.y, post.z);
    float time = post.w ; 

    G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position, time );

    pp.momentum_direction.set(dirw.x, dirw.y ,dirw.z);

    G4ThreeVector pol ; 
    pol.set(polw.x, polw.y, polw.z );

    G4double weight = dirw.w ;  // usually 1.0
    G4double wavelength = polw.w ;  // nm 
    G4double energy = h_Planck*c_light/wavelength ;
    pp.energy = energy ;

    if(m_sourcedbg && pho_index < 10) 
    LOG(info) << "CInputPhotonSource::convertPhoton"
              << " pho_index " << std::setw(6) << pho_index 
              << " nm " << wavelength
              << " wt " << weight
              << " time " << time
              << " pos (" 
              << " " << pp.position.x()
              << " " << pp.position.y()
              << " " << pp.position.z()
              << " )"
              << " dir ("
              << " " << pp.momentum_direction.x()
              << " " << pp.momentum_direction.y()
              << " " << pp.momentum_direction.z()
              << " )"
              << " pol ("
              << " " << pol.x()
              << " " << pol.y()
              << " " << pol.z()
              << " )"
              ;

    G4double mass = m_definition->GetPDGMass();
    G4double charge = m_definition->GetPDGCharge();

    G4PrimaryParticle* particle = new G4PrimaryParticle(m_definition);
    particle->SetKineticEnergy(pp.energy );
    particle->SetMass( mass );
    particle->SetMomentumDirection( pp.momentum_direction );
    particle->SetCharge( charge );
    particle->SetPolarization(pol); 
    particle->SetWeight(weight);

    vertex->SetPrimary(particle);

    return vertex ; 
}


void CInputPhotonSource::GeneratePrimaryVertex(G4Event *evt) 
{
    unsigned n = m_tranche->tranche_size(m_gpv_count) ; 
    SetNumberOfParticles(n);
    assert( m_num == int(n) );
	for (G4int i = 0; i < m_num; i++) 
    {
        unsigned pho_index = m_tranche->global_index( m_gpv_count,  i) ;
        G4PrimaryVertex* vertex = convertPhoton(pho_index);
        evt->AddPrimaryVertex(vertex);
        collectPrimary(vertex);
	}
    m_gpv_count++ ; 
}


