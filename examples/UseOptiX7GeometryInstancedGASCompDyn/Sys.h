#pragma once

struct Sys
{
    static float unsigned_as_float( unsigned u ) ;
    static unsigned float_as_unsigned( float f  ) ;
};


inline float Sys::unsigned_as_float( unsigned u ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.u = u  ;   
    return uif.f ; 
}

inline unsigned Sys::float_as_unsigned( float f  ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.f = f  ;   
    return uif.u ; 
}



