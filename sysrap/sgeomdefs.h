#pragma once

#include "geomdefs.hh"

struct sgeomdefs
{
    static constexpr const char* kOutside_ = "kOutside" ; 
    static constexpr const char* kSurface_ = "kSurface" ; 
    static constexpr const char* kInside_  = "kInside" ; 
    static const char* EInside_( EInside in ); 
};

inline const char* sgeomdefs::EInside_( EInside in )
{
    const char* s = nullptr ; 
    switch(in)
    {
        case kOutside: s = kOutside_ ; break ; 
        case kSurface: s = kSurface_ ; break ; 
        case kInside:  s = kInside_  ; break ; 
    }
    return s ; 
}

