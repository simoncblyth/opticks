#pragma once
/**
U4VPrimaryGenerator.h
=======================

Implemented header only to allow isolating use
of MOCK_CURAND to the test executable only, and 
not the library.


**/

struct sphoton ; 
class G4PrimaryVertex ; 
class G4Event ; 

struct U4VPrimaryGenerator
{
    template<typename P> 
    static void GetPhotonParam( 
         G4ThreeVector& position_mm, G4double& time_ns, 
         G4ThreeVector& direction,  G4double& wavelength_nm,
         G4ThreeVector& polarization, const P& p ); 
    
    template<typename P> 
    static G4PrimaryVertex* MakePrimaryVertexPhoton( const P& p); 

    static void GeneratePrimaries(G4Event *event); 
};


#include <cassert>

#include "G4Event.hh"
#include "G4OpticalPhoton.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "SGenerate.h"  // needs MOCK_CURAND

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "NP.hh"


template<typename P>
inline void U4VPrimaryGenerator::GetPhotonParam( 
     G4ThreeVector& position_mm, G4double& time_ns, 
     G4ThreeVector& direction,  G4double& wavelength_nm,
     G4ThreeVector& polarization, const P& p )
{
     position_mm.set(p.pos.x, p.pos.y, p.pos.z);
     time_ns = p.time ; 

     direction.set(p.mom.x, p.mom.y, p.mom.z ); 
     polarization.set(p.pol.x, p.pol.y, p.pol.z ); 
     wavelength_nm = p.wavelength ;   
}


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

Notice that there are no G4Track in sight here, so there is no 
way to annotate the tracks with *spho* labels.  

**/

inline void U4VPrimaryGenerator::GeneratePrimaries(G4Event* event)
{
    NP* ph = SGenerate::GeneratePhotons(); 
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
            const sphoton& p = pp[i]; 
            //if(i < 10) std::cout << "U4VPrimaryGenerator::GeneratePrimaries p.desc " << p.desc() << std::endl ; 
            G4PrimaryVertex* vertex = MakePrimaryVertexPhoton<sphoton>( p ); 
            event->AddPrimaryVertex(vertex);
        } 
    }
    else if( ph->ebyte == 8 )
    {
        sphotond* pp = (sphotond*)ph->bytes() ; 
        for(int i=0 ; i < ph->shape[0] ; i++)
        {
            const sphotond& p = pp[i]; 
            G4PrimaryVertex* vertex = MakePrimaryVertexPhoton<sphotond>( p ); 
            event->AddPrimaryVertex(vertex);
        } 
    }
}


