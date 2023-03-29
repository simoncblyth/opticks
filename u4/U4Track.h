#pragma once

#include <string>
class G4Track ; 

struct U4Track
{
    static G4Track* MakePhoton(); 
    static int Id(const G4Track* track); 
    static void SetId(G4Track* track, int id); 
    static bool IsOptical(const G4Track* track); 
    static void SetStopAndKill(const G4Track* track); 

    template<typename T>
    static std::string Desc(const G4Track* track); 

    // pass-thru to STrackInfo methods Label and LabelRef removed 

    template<typename T>
    static void SetFabricatedLabel(const G4Track* track); 

};


#include <sstream>
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

/**
U4Track::Id
-------------

0-based Id (unlike original G4Track::GetTrackID which is 1-based)

**/

inline int U4Track::Id(const G4Track* track)
{
    return track->GetTrackID() - 1 ;   
}

/**
U4Track::SetId
----------------

NB *id* is 0-based but Geant4 uses a 1-based TrackId  

**/
inline void U4Track::SetId(G4Track* track, int id)
{
    track->SetTrackID( id + 1 ); 
}

inline bool U4Track::IsOptical(const G4Track* track)
{
    G4ParticleDefinition* particle = track->GetDefinition(); 
    return particle == G4OpticalPhoton::OpticalPhotonDefinition() ; 
}

inline void U4Track::SetStopAndKill(const G4Track* track)
{
    G4Track* track_ = const_cast<G4Track*>(track); 
    track_->SetTrackStatus(fStopAndKill);
}


#ifdef WITH_CUSTOM4
#include "C4TrackInfo.h"
#else
#include "STrackInfo.h"
#endif

template<typename T>
inline void U4Track::SetFabricatedLabel(const G4Track* track)
{
    int trackID = Id(track) ; 
    assert( trackID >= 0 );  
    T fab = T::Fabricate(trackID); 
    G4Track* _track = const_cast<G4Track*>(track);  

#ifdef WITH_CUSTOM4
    C4TrackInfo<T>::Set(_track, fab );        
#else
    STrackInfo<T>::Set(_track, fab );        
#endif
}

template<typename T>
inline std::string U4Track::Desc(const G4Track* track)
{
#ifdef WITH_CUSTOM4
    T* label = C4TrackInfo<T>::GetRef(track); 
#else
    T* label = STrackInfo<T>::GetRef(track); 
#endif

    std::stringstream ss ; 
    ss << "U4Track::Desc"
       << " Id " << std::setw(5) << Id(track)
       << " Op " << std::setw(1) << IsOptical(track)
       << " label  " << ( label ? label->desc() : "-" ) 
       ;

    std::string s = ss.str(); 
    return s ; 
}

