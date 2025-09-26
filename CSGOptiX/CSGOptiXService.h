#pragma once

/**
CSGOptiXService.h
==================

The CSGOptiXService.h struct provides a very high level C++ API to CSGOptiX
functionality that is exposed to python within the "opticks_CSGOptiX" extension module.
This is implemented using nanobind and NP_nanobind.h which does NP <-> numpy conversions.

Use from python with::

    import opticks_CSGOptiX as cx

    svc = cx.CSGOptiXService()

    gs = ... # access gensteps

    ht = svc.simulate(gs)

    ## do something with hits

**/

#include "ssys.h"
#include "SEvt.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "NP.hh"

struct CSGOptiXService
{
    static CSGOptiXService* INSTANCE ;
    static CSGOptiXService* Get();
    static NP* Simulate(NP* gs, int eventID );

    int         level ;
    SEvt*       evt ;
    CSGFoundry* fd ;
    CSGOptiX*   cx ;

    CSGOptiXService();
    NP* simulate(NP* gs, int eventID );
    std::string desc() const ;
};


CSGOptiXService* CSGOptiXService::INSTANCE = nullptr ;

CSGOptiXService* CSGOptiXService::Get()
{
    if(!INSTANCE) new CSGOptiXService ;
    assert(INSTANCE);
    return INSTANCE ;
}

NP* CSGOptiXService::Simulate( NP* gs, int eventID )
{
    CSGOptiXService* svc = Get();
    return svc->simulate(gs, eventID );
}


inline CSGOptiXService::CSGOptiXService()
    :
    level(ssys::getenvint("CSGOptiXService_level",0)),
    evt(SEvt::Create(SEvt::EGPU)),
    fd(CSGFoundry::Load()),
    cx(CSGOptiX::Create(fd))
{
    INSTANCE = this ;
    std::cout << desc() ;
}


inline NP* CSGOptiXService::simulate( NP* gs, int eventID )
{
    if(level > 0) std::cout << "[CSGOptiXService::simulate gs " << ( gs ? gs->sstr() : "-" ) << "\n" ;

    NP* ht = cx->simulate(gs, eventID );

    if(level > 0) std::cout << "]CSGOptiXService::simulate ht " << ( ht ? ht->sstr() : "-" ) << "\n" ;
    return ht ;
}

inline std::string CSGOptiXService::desc() const
{
    std::stringstream ss ;
    ss << "-CSGOptiXService::desc"
       << " evt " << ( evt ? "YES" : "NO " )
       << " fd " << ( fd ? "YES" : "NO " )
       << " cx " << ( cx ? "YES" : "NO " )
       << "\n"
       ;

    std::string str = ss.str() ;
    return str ;
}


