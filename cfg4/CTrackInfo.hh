#pragma once

#include "G4VUserTrackInformation.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CTrackInfo
============

**/

struct CFG4_API CTrackInfo : public G4VUserTrackInformation
{
    CTrackInfo( int photon_record_id_ )
        :
        photon_record_id(photon_record_id_)
    {
    }

    int photon_record_id ;  
};

#include "CFG4_TAIL.hh"
