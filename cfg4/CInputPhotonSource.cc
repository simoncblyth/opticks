#include "CFG4_BODY.hh"
#include <sstream>
#include <cmath>

// sysrap-
#include "STranche.hh"

// npy-
#include "NPY.hpp"
#include "NPho.hpp"
#include "GLMFormat.hpp"
#include "GenstepNPY.hpp"

#include "Opticks.hh"
#include "GConstant.hh"

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

/**
CInputPhotonSource::convertPhoton
----------------------------------

Converts n_pho input photon at index pho_index into a G4PrimaryVertex
with the currently defined particle m_definition.


**/

G4PrimaryVertex* CInputPhotonSource::convertPhoton(unsigned pho_index)
{
    part_prop_t& pp = m_pp.Get();

    glm::vec4 post = m_pho->getPositionTime(pho_index) ; 
    glm::vec4 dirw = m_pho->getDirectionWeight(pho_index) ; 
    glm::vec4 polw = m_pho->getPolarizationWavelength(pho_index) ; 

    pp.position.set(post.x, post.y, post.z);
    float time = post.w ; 

    G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position, time );

    //pp.momentum_direction.set(dirw.x, dirw.y ,dirw.z);

    G4ThreeVector momdir(dirw.x, dirw.y ,dirw.z); 
    momdir = momdir.unit() ;  
    pp.momentum_direction.set(momdir.x(), momdir.y() ,momdir.z());

    // double precision normalize to avoid notes/issues/G4_barfs_tboolean_sphere_emitter.rst


    G4ThreeVector pol ; 
    pol.set(polw.x, polw.y, polw.z );

    G4double weight = dirw.w ;  // usually 1.0


    G4double wavelength_nm = polw.w ;  // nm 
    G4double energy_eV = GConstant::hc_eVnm/wavelength_nm ;   // GConstant::hc_eVnm = 1239.841875  (see GConstantTest) 
    G4double energy_MeV = energy_eV*1e-6 ;   

    // cf CMPT::addProperty   

    G4double wavelength = double(polw.w)*nm ;  
    G4double energy = h_Planck*c_light/wavelength ;  

    assert( fabs(energy-energy_MeV) < 1e-12 );


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




/**
CInputPhotonSource::GeneratePrimaryVertex
------------------------------------------

Repeated calls to this for each Geant4 event hook up the configured
max of photons per event for all but the last tranche, which just 
does the remainder.  The G4PrimaryVertex created for each input photon
as well as being added to the G4Event is collected using the base class
CSource::collectPrimary.

**/

void CInputPhotonSource::GeneratePrimaryVertex(G4Event *evt) 
{
    unsigned n = m_tranche->tranche_size(m_gpv_count) ; 
    SetNumberOfParticles(n);
    assert( m_num == int(n) );

    LOG(info) << "CInputPhotonSource::GeneratePrimaryVertex"
              << " n " << n 
               ;

	for (G4int i = 0; i < m_num; i++) 
    {
        unsigned pho_index = m_tranche->global_index( m_gpv_count,  i) ;

        G4PrimaryVertex* vertex = convertPhoton(pho_index); 
        // just straight convert of the photon pulled out the buffer

        evt->AddPrimaryVertex(vertex);
        collectPrimary(vertex);
	}
    m_gpv_count++ ; 
}


std::string CInputPhotonSource::desc() const 
{
    std::stringstream ss ; 
    ss << "CInputPhotonSource"
       ;
    return ss.str();
}


