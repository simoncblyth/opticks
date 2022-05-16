#pragma once


class G4Track ; 

struct U4Track
{
    static G4Track* MakePhoton(); 
};

#include "G4Track.hh"
#include "G4OpticalPhoton.hh"

inline G4Track* U4Track::MakePhoton()
{
    G4ParticleMomentum momentum(0., 0., 1.); 
    G4DynamicParticle* particle = new G4DynamicParticle(G4OpticalPhoton::Definition(),momentum);
    particle->SetPolarization(0., 1., 0. );  

    G4double time(0.); 

    G4ThreeVector position(0., 0., 0.); 

    G4Track* track = new G4Track(particle,time,position);
    return track ; 
}


