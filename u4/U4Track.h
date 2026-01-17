#pragma once
/**
U4Track.h
===========

Several static methods take templated photon label types such as::

    spho.h
    C4Pho.h (same impl as spho.h but from CUSTOM4)

**/



#include <string>
class G4Track ;

struct U4Track
{
    static G4Track* MakePhoton();
    static int Id(const G4Track* track);
    static void SetId(G4Track* track, int id);
    static bool IsOptical(const G4Track* track);
    static void SetStopAndKill(const G4Track* track);

    static std::string Desc(const G4Track* track);
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
#include "C4Pho.h"
#include "C4TrackInfo.h"
#else
#include "STrackInfo.h"
#endif

inline void U4Track::SetFabricatedLabel(const G4Track* track)
{
    int trackID = Id(track) ;
    assert( trackID >= 0 );
    G4Track* _track = const_cast<G4Track*>(track);

#ifdef WITH_CUSTOM4_OLD
    C4Pho fab = C4Pho::Fabricate(trackID);
    C4TrackInfo<C4Pho>::Set(_track, fab );
#elif WITH_CUSTOM4
    C4Pho fab = C4Pho::Fabricate(trackID);
    C4TrackInfo::Set(_track, fab );
#else
    spho fab = spho::Fabricate(trackID);
    STrackInfo::Set(_track, fab );
#endif
}

inline std::string U4Track::Desc(const G4Track* track)
{
#ifdef WITH_CUSTOM4_OLD
    C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track);
#elif WITH_CUSTOM4
    C4Pho* label = C4TrackInfo::GetRef(track);
#else
    spho* label = STrackInfo::GetRef(track);
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

