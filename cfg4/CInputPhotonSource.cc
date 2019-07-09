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
#include "OpticksFlags.hh"
#include "GConstant.hh"

// cfg4-
#include "CInputPhotonSource.hh"
#include "CEventInfo.hh"
#include "CRecorder.hh"

// g4-
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4TrackingManager.hh"
#include "G4Track.hh"

#include "PLOG.hh"

unsigned CInputPhotonSource::getNumG4Event() const { return m_tranche->num_tranche ; }
unsigned CInputPhotonSource::getNumPhotonsPerG4Event() const { return m_numPhotonsPerG4Event ; }

CInputPhotonSource::CInputPhotonSource(Opticks* ok, NPY<float>* input_photons, GenstepNPY* gsnpy )  
    :
    CSource(ok),
    m_sourcedbg(ok->isDbgSource()),
    m_pho(new NPho(input_photons)),
    m_gsnpy(gsnpy), 
    m_numPhotonsPerG4Event(m_gsnpy->getNumPhotonsPerG4Event()),
    m_numPhotons(m_pho->getNumPhotons()),
    m_tranche(new STranche(m_numPhotons,m_numPhotonsPerG4Event)),
    m_gpv_count(0)
{
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
    glm::vec4 post = m_pho->getPositionTime(pho_index) ; 
    glm::vec4 dirw = m_pho->getDirectionWeight(pho_index) ; 
    glm::vec4 polw = m_pho->getPolarizationWavelength(pho_index) ; 

    G4ThreeVector position(post.x, post.y, post.z);
    G4double time = post.w ; 

    G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time );

    G4double weight = dirw.w ; 
    assert( weight == 1.0 ) ; 

    G4ThreeVector direction(dirw.x, dirw.y ,dirw.z); 
    direction = direction.unit() ;  
    // double precision normalize to avoid notes/issues/G4_barfs_tboolean_sphere_emitter.rst

    G4ThreeVector polarization(polw.x, polw.y, polw.z);   //

    G4double wavelength_nm = polw.w ;  // nm 
    G4double energy_eV = GConstant::hc_eVnm/wavelength_nm ;   // GConstant::hc_eVnm = 1239.841875  (see GConstantTest) 
    G4double energy_MeV = energy_eV*1e-6 ;   

    // cf CMPT::addProperty   
    G4double wavelength = double(polw.w)*nm ;  
    G4double kineticEnergy = h_Planck*c_light/wavelength ;  

    assert( fabs(kineticEnergy-energy_MeV) < 1e-12 );


    if(m_sourcedbg && pho_index < 10) 
    LOG(info)
        << " pho_index " << std::setw(6) << pho_index 
        << " nm " << wavelength
        << " wt " << weight
        << " time " << time
        << " pos (" 
        << " " << position.x()
        << " " << position.y()
        << " " << position.z()
        << " )"
        << " dir ("
        << " " << direction.x()
        << " " << direction.y()
        << " " << direction.z()
        << " )"
        << " pol ("
        << " " << polarization.x()
        << " " << polarization.y()
        << " " << polarization.z()
        << " )"
        ;


    G4ParticleDefinition* definition = G4OpticalPhoton::Definition(); 

    G4PrimaryParticle* particle = new G4PrimaryParticle(definition);
    particle->SetKineticEnergy(kineticEnergy );
    particle->SetMomentumDirection( direction );
    particle->SetPolarization(polarization); 
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
    OK_PROFILE("_CInputPhotonSource::GeneratePrimaryVertex"); 
      
    unsigned num_photons = m_tranche->tranche_size(m_gpv_count) ; 

    unsigned event_gencode = TORCH ;   // no 1-based ffs indexable space for a new code, so reuse TORCH 
    evt->SetUserInformation( new CEventInfo(event_gencode)) ;

    LOG(info)
        << " num_photons " << num_photons
        << " gpv_count " << m_gpv_count
        << " event_gencode " << event_gencode
        << " : " << OpticksFlags::Flag(event_gencode)
        ; 

	for (unsigned i = 0; i < num_photons ; i++) 
    {
        unsigned pho_index = m_tranche->global_index( m_gpv_count,  i) ;

        G4PrimaryVertex* vertex = convertPhoton(pho_index); 
        // just straight convert of the photon pulled out the buffer

        evt->AddPrimaryVertex(vertex);
        collectPrimaryVertex(vertex);
	}
    m_gpv_count++ ; 
    OK_PROFILE("CInputPhotonSource::GeneratePrimaryVertex"); 
}


std::string CInputPhotonSource::desc() const 
{
    std::stringstream ss ; 
    ss << "CInputPhotonSource"
       ;
    return ss.str();
}


