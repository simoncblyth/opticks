#pragma once

/**
CSGOptiXService.h
==================

Thinking about a very high level API to expose to python with nanobind.

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
    NP* simulate(const NP* gs);
};


inline CSGOptiXService::CSGOptiXService()
    :
    evt(SEvt::Create(SEvt::EGPU)),
    fd(CSGFoundry::Load()),
    cx(CSGOptiX::Create(fd))
{
}

inline NP* CSGOptiXService::simulate( const NP* gs )
{
    NP* ht = gs->copy();
    return ht ;
}









