#pragma once

#include <string>
#include "CFG4_API_EXPORT.hh"

struct CFG4_API CPho 
{
    unsigned gs ;   // 0-based genstep index within the event,  from this can lookup CGenstep info
    unsigned ix ;   // 0-based photon index within the genstep
    unsigned id ;   // 0-based photon idenity within the event 
    bool     re ;   // reemission photon 

    CPho(unsigned gs, unsigned ix, unsigned id, bool re);
    std::string desc() const ;
};



