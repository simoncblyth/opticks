#pragma once

#include "G4VUserTrackInformation.hh"


/**
TrackInfo
============

**/

struct TrackInfo : public G4VUserTrackInformation
{
    TrackInfo( int photon_record_id_ )
        :
        photon_record_id(photon_record_id_)
    {
    }

    int photon_record_id ;  
};

