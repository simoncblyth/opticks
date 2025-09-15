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

#include "SEvt.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "NP.hh"

struct CSGOptiXService
{
    SEvt*       evt ;
    CSGFoundry* fd ;
    CSGOptiX*   cx ;

    CSGOptiXService();
    NP* simulate(const NP* gs) const ;
    std::string desc() const ;
};

inline CSGOptiXService::CSGOptiXService()
    :
    evt(SEvt::Create(SEvt::EGPU)),
    fd(CSGFoundry::Load()),
    cx(CSGOptiX::Create(fd))
{
    std::cout << desc() ;
}


inline NP* CSGOptiXService::simulate( const NP* gs ) const
{
    std::cout << "[CSGOptiXService::simulate gs " << ( gs ? gs->sstr() : "-" ) << "\n" ;

    NP* ht = gs->copy();

    std::cout << "]CSGOptiXService::simulate ht " << ( ht ? ht->sstr() : "-" ) << "\n" ;
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


