#pragma once

/**
CSGOptiXService.h
==================

Thinking about a very high level API to expose to python with nanobind.

**/

#include <string>

struct Dog
{
    std::string name;
    std::string bark() const ;
};
inline std::string Dog::bark() const
{
    return name + ": woof!" ;
}



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



inline NP* CSGOptiXService::simulate( const NP* gs )
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




