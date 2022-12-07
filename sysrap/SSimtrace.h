#pragma once
/**
SSimtrace : aiming to replace X4Simtrace
=============================================

**/
#include <cstring>
#include "plog/Severity.h"
#include "sframe.h"

class G4VSolid ; 
struct SEvt ; 

struct SSimtrace
{
    static constexpr const plog::Severity LEVEL = info ; 
    static void Scan( const G4VSolid* solid, const char* base=nullptr ); 

    const G4VSolid* solid ; 
    SEvt* evt ; 
    sframe frame ;  

    SSimtrace(); 
    ~SSimtrace(); 

    void setSolid(const G4VSolid* solid); 
    void simtrace(); 
    void saveEvent(const char* base); 
};

#include "SEvt.hh"
#include "SEventConfig.hh"
#include "ssolid.h"
#include "SLOG.hh"


inline void SSimtrace::Scan(const G4VSolid* solid, const char* base )
{
    SSimtrace t ; 
    t.setSolid(solid); 
    t.simtrace(); 

    G4String soname = solid->GetName(); 
    t.evt->setReldir(soname.c_str()); 

    t.saveEvent(base) ; 
}


inline SSimtrace::SSimtrace()
    :
    solid(nullptr), 
    evt(nullptr)
{
}

inline SSimtrace::~SSimtrace()
{
    delete evt ; 
}

inline void SSimtrace::setSolid(const G4VSolid* solid_)
{
    LOG(LEVEL) ; 
    solid = solid_ ; 
    ssolid::GetCenterExtent(frame.ce, solid );   
}

/**
SSimtrace::simtrace
---------------------

SEvt::setFrame

1. creates gensteps with SFrameGenstep::MakeCenterExtentGensteps and adds them to SEvt
2. As frame.is_hostside_simtrace also generates simtrace "photons" 

**/

inline void SSimtrace::simtrace()
{
    SEventConfig::SetRGModeSimtrace();
    frame.set_hostside_simtrace();  

    evt = new SEvt ; 
    evt->setFrame(frame);   
    // in RGModeSimtrace SEvt::setFrame adds simtrace gensteps 
    // configured via envvars

    LOG(LEVEL) << " evt.simtrace.size " << evt->simtrace.size() ; 

    bool dump = false ; 
    for(unsigned i=0 ; i < evt->simtrace.size() ; i++)
    {
         quad4& p = evt->simtrace[i] ; 
         ssolid::Simtrace(p, solid, dump);  
    }
}

inline void SSimtrace::saveEvent(const char* base)
{
    LOG(LEVEL) ; 
    evt->save(base);  
}

