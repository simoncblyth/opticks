#pragma once
/**
X4OpticalSurfaceFinish.hh
===========================

Basis G4OpticalSurfaceFinish enum { polished, polishedfrontpainted, ... }
is defined in G4OpticalSurface.hh

Note that the 6 finish types are only the first of 30 or so.
Only finish types used by active geometries have any hope of being supported. 

**/

#include <cstring>
#include <cassert>
#include "G4OpticalSurface.hh"

struct X4OpticalSurfaceFinish
{
    static const char* Name(unsigned finish); 
    static unsigned   Finish(const char* name);
   
    static constexpr const char* polished_             = "polished" ;              // smooth perfectly polished surface
    static constexpr const char* polishedfrontpainted_ = "polishedfrontpainted" ;  // smooth top-layer (front) paint
    static constexpr const char* polishedbackpainted_  = "polishedbackpainted"  ;  // same is 'polished' but with a back-paint
    static constexpr const char* ground_               = "ground" ;                // rough surface
    static constexpr const char* groundfrontpainted_   = "groundfrontpainted" ;    // rough top-layer (front) paint
    static constexpr const char* groundbackpainted_    = "groundbackpainted" ;     // same as 'ground' but with a back-paint
}; 

inline const char* X4OpticalSurfaceFinish::Name(unsigned finish)
{
    const char* n = nullptr ; 
    switch(finish)
    {
        case polished:              n = polished_             ; break ;  
        case polishedfrontpainted:  n = polishedfrontpainted_ ; break ;  
        case polishedbackpainted:   n = polishedbackpainted_  ; break ;  
        case ground:                n = ground_               ; break ;  
        case groundfrontpainted:    n = groundfrontpainted_   ; break ;  
        case groundbackpainted:     n = groundbackpainted_    ; break ;  
    }

    assert(n); 
    return n ; 
}

inline unsigned X4OpticalSurfaceFinish::Finish(const char* name)
{
    assert(name); 
    unsigned finish = 0 ; 
    if(strcmp(name, polished_)==0)             finish = polished ; 
    if(strcmp(name, polishedfrontpainted_)==0) finish = polishedfrontpainted ; 
    if(strcmp(name, polishedbackpainted_)==0)  finish = polishedbackpainted ; 
    if(strcmp(name, ground_)==0)               finish = ground ; 
    if(strcmp(name, groundfrontpainted_)==0)   finish = groundfrontpainted ; 
    if(strcmp(name, groundbackpainted_)==0)    finish = groundbackpainted ; 

    bool match = strcmp( Name(finish), name ) == 0 ;  
    assert( match ); 

    return finish ; 
}

