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

void X4Simtrace::simtrace()
{
    SEventConfig::SetRGModeSimtrace();
    evt = new SEvt ; 
    evt->setFrame(frame);  // creates gensteps with SFrameGenstep::MakeCenterExtentGensteps and adds them 
    LOG(LEVEL) ; 
}

void X4Simtrace::saveEvent()
{
    LOG(LEVEL) ; 
}


