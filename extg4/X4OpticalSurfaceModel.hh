#pragma once
/**
X4OpticalSurfaceModel.hh : string consts for G4OpticalSurfaceModel   { glisur, unified, ... }
=================================================================================================

Basis G4OpticalSurfaceModel enum { glisur, unified, ... } is defined in G4OpticalSurface.hh

Note that support is only being attempted for glisur and unified

**/

#include <cstring>
#include <cassert>
#include "G4OpticalSurface.hh"

struct X4OpticalSurfaceModel
{
    static const char* Name(unsigned model); 
    static unsigned    Model(const char* name);

    static constexpr const char* glisur_    = "glisur" ;   // original GEANT3 model
    static constexpr const char* unified_   = "unified" ;  // UNIFIED model
    static constexpr const char* LUT_       = "LUT" ;      // Look-Up-Table model
    static constexpr const char* DAVIS_     = "DAVIS" ;    // DAVIS model
    static constexpr const char* dichroic_  = "dichroic" ; // dichroic filter 
};

inline const char* X4OpticalSurfaceModel::Name(unsigned model)
{
    const char* n = nullptr ; 
    switch(model)
    {
        case glisur:    n = glisur_      ; break ; 
        case unified:   n = unified_     ; break ; 
        case LUT:       n = LUT_         ; break ; 
        case DAVIS:     n = DAVIS_       ; break ; 
        case dichroic:  n = dichroic_    ; break ; 
    }
    return n ; 
}

inline unsigned X4OpticalSurfaceModel::Model(const char* name)
{
    assert(name); 
    unsigned model = glisur ; 
    if(strcmp(name,glisur_)== 0  )    model = glisur ; 
    if(strcmp(name,unified_)== 0  )   model = unified ; 
    if(strcmp(name,LUT_)== 0  )       model = LUT ; 
    if(strcmp(name,DAVIS_)== 0  )     model = DAVIS ; 
    if(strcmp(name,dichroic_)== 0  )  model = dichroic ; 
    return model ; 
}


