#pragma once

#include <string>
#include "plog/Severity.h"
#include "G4VUserTrackInformation.hh"
#include "CGenstep.hh"
#include "CPho.hh"
#include "CFG4_API_EXPORT.hh"

/**
CPhotonInfo 
=============

id_==-1(default)
    photon id() --> gs.offset + ix_ 

    Assigns original photon id based on the genstep offset and index
    
id_>-1
    photon id() --> id_  

    Used for passing original photon id down thru RE-generations

**/

struct CFG4_API CPhotonInfo : public G4VUserTrackInformation
{
    static const plog::Severity LEVEL ; 

    static CPho         Get(const G4Track* optical_photon_track, bool when_unlabelled_fabricate_trackid_photon ); 
    static CPhotonInfo* MakeScintillation(const CGenstep& gs, unsigned i, const CPho& ancestor ); 
    static CPhotonInfo* MakeCerenkov(     const CGenstep& gs, unsigned i ) ; 

    CPho pho ;   

    CPhotonInfo(const CPho& _pho );
    CPhotonInfo(const CGenstep& gs, unsigned ix_ , unsigned gn_, int id_ );
    virtual ~CPhotonInfo(); 

    unsigned gs() const ; // 0-based genstep index within the event
    unsigned ix() const ; // 0-based photon index within the genstep
    unsigned id() const ; // 0-based absolute photon identity index within the event 
    unsigned gn() const ; // 0-based generation index, incremented at each reemission

    G4String*   type() const ; 
    std::string desc() const ;
};


