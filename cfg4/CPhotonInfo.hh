#pragma once

#include <string>
#include "plog/Severity.h"
#include "G4VUserTrackInformation.hh"
#include "CGenstep.hh"
#include "CPho.hh"
#include "CFG4_API_EXPORT.hh"

struct CFG4_API CPhotonInfo : public G4VUserTrackInformation
{
    static const plog::Severity LEVEL ; 
    CPho pho ;   

    CPhotonInfo(const CGenstep& gs, unsigned ix_ , bool re_ );
    virtual ~CPhotonInfo(); 

    G4String*   type() const ; 
    std::string desc() const ;
};


