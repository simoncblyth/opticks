#pragma once
/**
SEvt.h
========

Replacing cfg4/CGenstepCollector

HMM: gs vector of sgs provides summary of the full genstep, 
changing the first quad of the genstep to hold this summary info 
would avoid the need for the summary vector and mean the genstep 
index and photon offset info was available on device.

Header of full genstep has plenty of spare bits to squeeze in
index and photon offset in addition to  gentype/trackid/matline/numphotons 

**/

#include <cassert>
#include <vector>
#include <string>
#include <sstream>

#include "scuda.h"
#include "squad.h"
#include "sgs.h"

struct SEvt
{
    static SEvt* INSTANCE ; 
    static SEvt* Get() ; 
    static sgs AddGenstep(const quad6& q); 

    std::vector<quad6> genstep ; 
    std::vector<sgs>   gs ; 

    SEvt(); 

    void clear() ; 
    unsigned getNumGenstep() const ; 
    unsigned getNumPhoton() const ; 
    sgs addGenstep(const quad6& q) ; 

    std::string desc() const ; 
};


SEvt* SEvt::INSTANCE = nullptr ; 
SEvt* SEvt::Get(){ return INSTANCE ; }

sgs SEvt::AddGenstep(const quad6& q)
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before SEvt::AddGenstep  " << std::endl ; 
    assert(INSTANCE); 
    return INSTANCE->addGenstep(q); 
}


inline SEvt::SEvt(){ INSTANCE = this ; }


inline void SEvt::clear()
{
    genstep.clear();
    gs.clear(); 
}

inline unsigned SEvt::getNumGenstep() const 
{
    assert( genstep.size() == gs.size() ); 
    return genstep.size() ; 
}

inline unsigned SEvt::getNumPhoton() const 
{
    unsigned tot = 0 ; 
    for(unsigned i=0 ; i < genstep.size() ; i++) tot += genstep[i].numphoton() ; 
    return tot ; 
}

inline sgs SEvt::addGenstep(const quad6& q)
{
    sgs s = {} ; 

    s.index = genstep.size() ;  // 0-based genstep index in event (actually since last reset)  
    s.photons = q.numphoton() ;  
    s.offset = getNumPhoton() ; // number of photons in event before this genstep  (actually single last reset)  
    s.gentype = q.gentype() ; 

    gs.push_back(s) ; 
    genstep.push_back(q) ; 

    return s ; 
}

inline std::string SEvt::desc() const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < getNumGenstep() ; i++) ss << gs[i].desc() << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}



