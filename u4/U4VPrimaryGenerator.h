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
    static G4PrimaryVertex* MakePrimaryVertexPhoton( const sphoton& p); 
    static void GeneratePrimaryVertex(G4Event *evt); 
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


inline G4PrimaryVertex* U4VPrimaryGenerator::MakePrimaryVertexPhoton( const sphoton& p)
{
    G4ThreeVector position(p.pos.x, p.pos.y, p.pos.z); 
    G4double time(p.time) ; 
    G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);

    G4ThreeVector direction( p.mom.x, p.mom.y, p.mom.z);   
    G4double kineticEnergy = h_Planck*c_light/(p.wavelength*nm) ; 
    G4ThreeVector polarization( p.pol.x, p.pol.y, p.pol.z );  

    G4PrimaryParticle* particle = new G4PrimaryParticle(G4OpticalPhoton::Definition());
    particle->SetKineticEnergy( kineticEnergy );
    particle->SetMomentumDirection( direction ); 
    particle->SetPolarization(polarization); 

    vertex->SetPrimary(particle);
    return vertex ; 
}

/**
U4VPrimaryGenerator::GeneratePrimaryVertex
-------------------------------------------

Notice that there are no G4Track in sight here, so there is no 
way to annotate the tracks with *spho* labels.  

**/

inline void U4VPrimaryGenerator::GeneratePrimaryVertex(G4Event* event)
{
    NP* ph = SGenerate::GeneratePhotons(); 
    if(ph == nullptr) std::cerr 
         << "U4VPrimaryGenerator::GeneratePrimaryVertex : FATAL : NO PHOTONS " << std::endl 
         << " compile with MOCK_CURAND to use SGenerate.h " << std::endl 
         ; 
    if(ph == nullptr) return ;  

    std::cout << " ph " << ( ph ? ph->sstr() : "-" ) << std::endl ;  
    sphoton* pp = (sphoton*)ph->bytes() ; 

    for(int i=0 ; i < ph->shape[0] ; i++)
    {
        const sphoton& p = pp[i]; 
        if(i < 10) std::cout << p.desc() << std::endl ; 

        G4PrimaryVertex* vertex = MakePrimaryVertexPhoton( p ); 
        event->AddPrimaryVertex(vertex);
    } 
}


