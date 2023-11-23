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
   T_CIRCLE,
   T_RECTANGLE,
   T_SPHERE_MARSAGLIA,
   T_SPHERE,
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
    static constexpr const char* T_CIRCLE_  = "circle" ;
    static constexpr const char* T_RECTANGLE_  = "rectangle" ;
    static constexpr const char* T_SPHERE_MARSAGLIA_  = "sphere_marsaglia" ;
    static constexpr const char* T_SPHERE_  = "sphere" ;

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
    if(strcmp(name,T_CIRCLE_)==0) type = T_CIRCLE ; 
    if(strcmp(name,T_RECTANGLE_)==0) type = T_RECTANGLE ; 
    if(strcmp(name,T_SPHERE_MARSAGLIA_)==0) type = T_SPHERE_MARSAGLIA ; 
    if(strcmp(name,T_SPHERE_)==0) type = T_SPHERE ; 
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
        case T_CIRCLE:  n = T_CIRCLE_ ; break ; 
        case T_RECTANGLE:  n = T_RECTANGLE_ ; break ; 
        case T_SPHERE_MARSAGLIA:  n = T_SPHERE_MARSAGLIA_ ; break ; 
        case T_SPHERE:  n = T_SPHERE_ ; break ; 
        default : break ; 
    }
    return n ; 
}
inline bool storchtype::IsValid(unsigned type)
{
    return type > T_UNDEF && type < T_LAST ; 
}

#endif


