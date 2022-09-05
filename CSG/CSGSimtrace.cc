#include "PLOG.hh"
#include "SEventConfig.hh"
#include "SOpticksResource.hh"
#include "SEvt.hh"
#include "SSim.hh"
#include "CSGFoundry.h"
#include "CSGSimtrace.hh"
#include "CSGQuery.h"
#include "CSGDraw.h"

const plog::Severity CSGSimtrace::LEVEL = PLOG::EnvLevel("CSGSimtrace", "DEBUG"); 

int CSGSimtrace::Preinit()    // static
{
    SEventConfig::SetRGModeSimtrace();
    return 0 ; 
}

CSGSimtrace::CSGSimtrace()
    :   
    prc(Preinit()),
    geom(ssys::getenvvar("GEOM", "nmskSolidMaskTail")),  
    sim(SSim::Create()),
    fd(CSGFoundry::Load()),
    evt(new SEvt),
    q(new CSGQuery(fd)),
    d(new CSGDraw(q,'Z'))
{
    d->draw("CSGSimtrace");
    frame.set_hostside_simtrace();  
    frame.ce = q->select_prim_ce ; 
    LOG(LEVEL) << " frame.ce " << frame.ce ; 

    evt->setFrame(frame);  
}

void CSGSimtrace::simtrace()
{
    LOG(LEVEL) << " evt.simtrace.size " << evt->simtrace.size() ; 
    for(unsigned i=0 ; i < evt->simtrace.size() ; i++)
    {
        quad4& p = evt->simtrace[i] ; 
        q->simtrace(p); 
    }
}

void CSGSimtrace::saveEvent()
{
    LOG(LEVEL) ; 
    evt->save();  
}

