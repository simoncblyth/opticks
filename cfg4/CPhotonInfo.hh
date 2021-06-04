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
    static int AncestralId(const G4Track* track, bool dump=false); 

    CPho pho ;   

    CPhotonInfo(const CGenstep& gs, unsigned ix_ , bool re_, int id_ );
    virtual ~CPhotonInfo(); 

    unsigned gs() const ; // 0-based genstep index within the event
    unsigned ix() const ; // 0-based photon index within the genstep
    unsigned id() const ; // 0-based absolute photon identity index within the event 
    bool     re() const ; // reemission flag 

    G4String*   type() const ; 
    std::string desc() const ;
};


