#pragma once

#include <string>
struct spho ; 
class G4Track ; 

struct U4Track
{
    static G4Track* MakePhoton(); 

    static int Id(const G4Track* track); 
    static bool IsOptical(const G4Track* track); 
    static std::string Desc(const G4Track* track); 
    static spho Label(const G4Track* track);   // returns placeholders when track has no label  
};


#include <sstream>
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "U4PhotonInfo.h"

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

int U4Track::Id(const G4Track* track)
{
    return track->GetTrackID() - 1 ;   
    // 0-based Id (unlike original G4Track::GetTrackID which is 1-based)
}

bool U4Track::IsOptical(const G4Track* track)
{
    G4ParticleDefinition* particle = track->GetDefinition(); 
    return particle == G4OpticalPhoton::OpticalPhotonDefinition() ; 
}

spho U4Track::Label(const G4Track* track)  // returns placeholders when track has no label 
{
    return U4PhotonInfo::Get(track) ;
}

std::string U4Track::Desc(const G4Track* track)
{
    spho sp = Label(track); 

    std::stringstream ss ; 
    ss << "U4Track::Desc"
       << " Id " << std::setw(5) << Id(track)
       << " Op " << std::setw(1) << IsOptical(track)
       << " sp " << sp.desc() 
       ;

    std::string s = ss.str(); 
    return s ; 
}




