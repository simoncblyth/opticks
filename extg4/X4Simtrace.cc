#include "SEvt.hh"
#include "SEventConfig.hh"

#include "X4Simtrace.hh"
#include "x4solid.h"
#include "PLOG.hh"


const plog::Severity X4Simtrace::LEVEL = PLOG::EnvLevel("X4Simtrace", "DEBUG"); 

void X4Simtrace::setSolid(const G4VSolid* solid_)
{
    solid = solid_ ; 
    LOG(LEVEL) ; 
    x4solid::GetCenterExtent(frame.ce, solid );   
}

/**
X4Simtrace::simtrace
---------------------

SEvt::setFrame

1. creates gensteps with SFrameGenstep::MakeCenterExtentGensteps and adds them to SEvt
2. As frame.is_hostside_simtrace also generates simtrace "photons" 

**/

void X4Simtrace::simtrace()
{
    SEventConfig::SetRGModeSimtrace();
    frame.set_hostside_simtrace();  

    evt = new SEvt ; 
    evt->setFrame(frame);  

    LOG(LEVEL) << " evt.simtrace.size " << evt->simtrace.size() ; 

    bool dump = false ; 
    for(unsigned i=0 ; i < evt->simtrace.size() ; i++)
    {
         quad4& p = evt->simtrace[i] ; 
         x4solid::Simtrace(p, solid, dump);  
    }

    evt->gather(); 
}

void X4Simtrace::saveEvent()
{
    LOG(LEVEL) ; 
    evt->save();  
}


