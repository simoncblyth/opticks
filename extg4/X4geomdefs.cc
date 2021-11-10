#include "X4geomdefs.hh"

const char* X4geomdefs::kOutside_ = "kOutside" ;       
const char* X4geomdefs::kSurface_ = "kSurface" ;       
const char* X4geomdefs::kInside_ = "kInside" ;       

const char* X4geomdefs::EInside_( EInside in )
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


