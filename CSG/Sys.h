#pragma once

struct Sys
{
    static float unsigned_as_float( unsigned u ) ;
    static unsigned float_as_unsigned( float f  ) ;

    static float int_as_float( int i ) ;
    static int float_as_int( float f  ) ;

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

inline float Sys::int_as_float( int i ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.i = i  ;   
    return uif.f ; 
}

inline int Sys::float_as_int( float f  ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.f = f  ;   
    return uif.i ; 
}



