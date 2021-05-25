
#pragma once

#include "G4VUserTrackInformation.hh"


/**
G4OpticksUserTrackInfo
===========================

**/

struct G4OpticksUserTrackInfo : public G4VUserTrackInformation
{
    G4OpticksUserTrackInfo( unsigned record_id_ , char gentype_ )  // gentype_ 'C' or 'S' 
        :   
        packed((record_id_ & 0x7fffffff) | unsigned(gentype_ == 'C') << 31 )   
    {   
    }   
    unsigned packed  ;   

    char gentype() const       { return ( packed & 0x80000000 ) ? 'C' : 'S' ;  }
    unsigned record_id() const { return ( packed & 0x7fffffff ) ; } 
};

