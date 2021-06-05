#pragma once

#include <string>
#include "CFG4_API_EXPORT.hh"

struct CFG4_API CPho 
{
    static const unsigned MISSING ; 

    unsigned gs ;   // 0-based genstep index within the event
    unsigned ix ;   // 0-based photon index within the genstep
    unsigned id ;   // 0-based photon identity index within the event 
    unsigned gn ;   // 0-based generation index incremented at each reemission 

    CPho(); 
    CPho(unsigned gs, unsigned ix, unsigned id, unsigned gn);
    bool is_missing() const ; 

    int get_gs() const ; // -1 when missing, otherwise : 0-based genstep index within the event
    int get_ix() const ; // -1 when missing, otherwise : 0-based photon index within the genstep
    int get_id() const ; // -1 when missing, otherwise : 0-based photon identity index within the event  
    int get_gn() const ; // -1 when missing, otherwise : 0-based generation index incremented at each reemission  


    CPho make_reemit() const ; 
    std::string desc() const ;
};


