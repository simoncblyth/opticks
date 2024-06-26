#pragma once
/**
U4VPrimaryGenerator.h
=======================

Implemented header only to allow isolating use
of MOCK_CURAND to the test executable only, and 
not the library.


**/

struct NP ; 
struct sphoton ; 
class G4PrimaryVertex ; 
class G4Event ; 
#include "U4_API_EXPORT.hh"

struct U4_API U4VPrimaryGenerator
{
    template<typename P> 
    static void GetPhotonParam( 
         G4ThreeVector& position_mm, G4double& time_ns, 
         G4ThreeVector& direction,  G4double& wavelength_nm,
         G4ThreeVector& polarization, const P& p ); 
    
    template<typename P> 
    static G4PrimaryVertex* MakePrimaryVertexPhoton( const P& p); 

    static constexpr const char* _DEBUG_GENIDX = "U4VPrimaryGenerator__GeneratePrimaries_From_Photons_DEBUG_GENIDX" ; 
    static void GeneratePrimaries_From_Photons(G4Event* event, const NP* ph ); 

};


#include <cassert>

#include "G4Event.hh"
#include "G4OpticalPhoton.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "ssys.h"
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "NP.hh"

/**
U4VPrimaryGenerator::GetPhotonParam
-------------------------------------

Photon parameters from Opticks photon types such as sphoton.h shuffled into Geant4 three vectors. 

**/

template<typename P>
inline void U4VPrimaryGenerator::GetPhotonParam( 
     G4ThreeVector& position_mm, G4double& time_ns, 
     G4ThreeVector& direction,  G4double& wavelength_nm,
     G4ThreeVector& polarization, const P& p )
{
     position_mm.set(p.pos.x, p.pos.y, p.pos.z);
     time_ns = p.time ; 

     direction.set(p.mom.x, p.mom.y, p.mom.z ); 
     direction = direction.unit();   

     polarization.set(p.pol.x, p.pol.y, p.pol.z ); 
     wavelength_nm = p.wavelength ;   
}


/**
U4VPrimaryGenerator::MakePrimaryVertexPhoton
----------------------------------------------

Converts Opticks photon type P (eg sphoton.h) into G4PrimaryVertex

1. shuffle photon param from P into G4ThreeVector, G4double
2. create G4PrimaryVertex and populate with the param obtained above 

**/

template<typename P>
inline G4PrimaryVertex* U4VPrimaryGenerator::MakePrimaryVertexPhoton( const P& p)
{
    G4ThreeVector position_mm ; 
    G4double time_ns  ; 
    G4ThreeVector direction ;   
    G4double wavelength_nm ; 
    G4ThreeVector polarization;   

    GetPhotonParam( position_mm, time_ns, direction, wavelength_nm, polarization, p ); 

    G4PrimaryVertex* vertex = new G4PrimaryVertex(position_mm, time_ns);
    G4double kineticEnergy = h_Planck*c_light/(wavelength_nm*nm) ; 
    G4PrimaryParticle* particle = new G4PrimaryParticle(G4OpticalPhoton::Definition());
    particle->SetKineticEnergy( kineticEnergy );
    particle->SetMomentumDirection( direction ); 
    particle->SetPolarization(polarization); 

    vertex->SetPrimary(particle);
    return vertex ; 
}

/**
U4VPrimaryGenerator::GeneratePrimaries
---------------------------------------

1. populates the G4Event argument with photon parameters from ph array tas G4PrimaryVertex

Notice that there are no G4Track in sight here, so there is no 
direct way to annotate the tracks with *spho* labels.  

To do so would have to arrange to catch the tracks at labelling stage 
in U4Recorder::PreUserTrackingAction_Optical where could
associate back to the originating array, that could be held in the 
SEvt as "input photons". This relies on Geant4 being consistent
in the way PrimaryVertex become G4Track ? I expect it will work 
in purely optical case. 

HMM: These *ph* are effectively input photons (even though generated from gensteps),
should associate as such in the SEvt to retain access to these

**/



inline void U4VPrimaryGenerator::GeneratePrimaries_From_Photons(G4Event* event, const NP* ph )
{
    int DEBUG_GENIDX = ssys::getenvint(_DEBUG_GENIDX, -1 ) ; 

    std::cout 
         << "U4VPrimaryGenerator::GeneratePrimaries_From_Photons" 
         << " ph " << ( ph ? ph->sstr() : "-" ) 
         << std::endl 
         << " " << _DEBUG_GENIDX << " : " << DEBUG_GENIDX << " (when +ve, only generate tht photon idx)" 
         << std::endl 
         ; 

    if(ph == nullptr) std::cerr 
         << "U4VPrimaryGenerator::GeneratePrimaries : FATAL : NO PHOTONS " << std::endl 
         << "compile with MOCK_CURAND to use SGenerate.h curand on CPU" << std::endl 
         ; 
    if(ph == nullptr) return ;  

    //std::cout << "U4VPrimaryGenerator::GeneratePrimaries" << " ph " << ( ph ? ph->brief() : "-" ) << std::endl ;  

    if( ph->ebyte == 4 )
    {
        sphoton* pp = (sphoton*)ph->bytes() ; 
        for(int i=0 ; i < ph->shape[0] ; i++)
        {
            if( DEBUG_GENIDX > -1 && i != DEBUG_GENIDX ) continue ; 
            const sphoton& p = pp[i]; 
            G4PrimaryVertex* vertex = MakePrimaryVertexPhoton<sphoton>( p ); 
            event->AddPrimaryVertex(vertex);
        } 
    }
    else if( ph->ebyte == 8 )
    {
        sphotond* pp = (sphotond*)ph->bytes() ; 
        for(int i=0 ; i < ph->shape[0] ; i++)
        {
            if( DEBUG_GENIDX > -1 && i != DEBUG_GENIDX ) continue ; 
            const sphotond& p = pp[i]; 
            G4PrimaryVertex* vertex = MakePrimaryVertexPhoton<sphotond>( p ); 
            event->AddPrimaryVertex(vertex);
        } 
    }
}

