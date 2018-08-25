#include <sstream>

#include "NPY.hpp"

#include "G4ParticleChange.hh"
#include "G4Track.hh"
#include "G4DynamicParticle.hh"
#include "G4OpticalPhoton.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4Version.hh"

#include "C4PhotonCollector.hh"
#include "CPhotonCollector.hh"
#include "PLOG.hh"

C4PhotonCollector::C4PhotonCollector() 
    :
    m_photon_collector(new CPhotonCollector)
{
    NPY<float>* photon = m_photon_collector->getPhoton();
    photon->setArrayContentVersion(G4VERSION_NUMBER);
}

void C4PhotonCollector::savePhotons(const char* path) const 
{
    LOG(info) << " path " << path ; 
    m_photon_collector->save(path); 
}

std::string C4PhotonCollector::desc() const 
{
    std::stringstream ss ; 
    ss << "C4PhotonCollector"
       << " " << m_photon_collector->desc()
       ;
    return ss.str(); 
}

NPY<float>*  C4PhotonCollector::getPhoton() const 
{
    return m_photon_collector->getPhoton() ; 
}


/**
C4PhotonCollector::collectSecondaryPhotons
---------------------------------------------

Structured in this keyhole surgery way in order that the above code 
can stay verbatim the same as the G4Cerenkov::PostStepDoIt generation loop.

**/

void C4PhotonCollector::collectSecondaryPhotons( const G4VParticleChange* pc, unsigned idx ) 
{
    G4int numberOfSecondaries = pc->GetNumberOfSecondaries(); 

    LOG(info) << " numberOfSecondaries " << numberOfSecondaries ; 


    for( G4int i=0 ; i < numberOfSecondaries ; i++)
    {
        G4Track* track =  pc->GetSecondary(i) ; 

        assert( track->GetParticleDefinition() == G4OpticalPhoton::Definition() );  

        const G4DynamicParticle* aCerenkovPhoton = track->GetDynamicParticle() ;

        const G4ThreeVector& photonMomentum = aCerenkovPhoton->GetMomentumDirection() ;   

        const G4ThreeVector& photonPolarization = aCerenkovPhoton->GetPolarization() ; 

        G4double kineticEnergy = aCerenkovPhoton->GetKineticEnergy() ;  

        G4double wavelength = h_Planck*c_light/kineticEnergy ; 

        const G4ThreeVector& aSecondaryPosition = track->GetPosition() ;

        G4double aSecondaryTime = track->GetGlobalTime() ;

        G4double weight = track->GetWeight() ; 

        int flags_x = idx ;  
        int flags_y = i ;  
        int flags_z = 0 ;  
        int flags_w = 0 ;  

        m_photon_collector->collectPhoton(
                 aSecondaryPosition.x()/mm,  
                 aSecondaryPosition.y()/mm,  
                 aSecondaryPosition.z()/mm,  
                 aSecondaryTime/ns, 

                 photonMomentum.x(),
                 photonMomentum.y(),
                 photonMomentum.z(),
                 weight,

                 photonPolarization.x(),
                 photonPolarization.y(),
                 photonPolarization.z(),
                 wavelength/nm,

                 flags_x, 
                 flags_y, 
                 flags_z, 
                 flags_w
        ); 

    } 
}




