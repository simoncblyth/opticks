#pragma once
/**
storchtype.h
=============

**/


enum {
   T_UNDEF,
   T_DISC,
   T_LINE,
   T_POINT,
   T_LAST
}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cstring>

struct storchtype
{
    static constexpr const char* T_UNDEF_ = "undef";
    static constexpr const char* T_DISC_  = "disc" ;
    static constexpr const char* T_LINE_  = "line" ;
    static constexpr const char* T_POINT_  = "point" ;

    static unsigned Type(const char* name); 
    static const char* Name(unsigned type); 
    static bool     IsValid(unsigned type); 
}; 

inline unsigned storchtype::Type(const char* name) 
{
    unsigned type = T_UNDEF ;
    if(strcmp(name,T_DISC_)==0) type = T_DISC ; 
    if(strcmp(name,T_LINE_)==0) type = T_LINE ; 
    if(strcmp(name,T_POINT_)==0) type = T_POINT ; 
    return type ; 
}

inline const char* storchtype::Name(unsigned type)
{
    const char* n = T_UNDEF_ ; 
    switch(type)
    {
        case T_UNDEF: n = T_UNDEF_ ; break ; 
        case T_DISC:  n = T_DISC_ ; break ; 
        case T_LINE:  n = T_LINE_ ; break ; 
        case T_POINT:  n = T_POINT_ ; break ; 
        default : break ; 
    }
    return n ; 
}
inline bool storchtype::IsValid(unsigned type)
{
    return type > T_UNDEF && type < T_LAST ; 
}

#endif


