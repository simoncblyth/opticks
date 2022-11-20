#pragma once
/**
storchtype.h
=============

**/


enum {
   T_UNDEF,
   T_DISC,
   T_LINE
}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cstring>

struct storchtype
{
    static constexpr const char* T_UNDEF_ = "undef";
    static constexpr const char* T_DISC_  = "disc" ;
    static constexpr const char* T_LINE_  = "line" ;

    static unsigned Type(const char* name); 
    static const char* Name(unsigned type); 
}; 

inline unsigned storchtype::Type(const char* name) 
{
    unsigned type = T_UNDEF ;
    if(strcmp(name,T_DISC_)==0) type = T_DISC ; 
    if(strcmp(name,T_LINE_)==0) type = T_LINE ; 
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
    }
    return n ; 
}
#endif


