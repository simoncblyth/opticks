#pragma once
/**
torchtype.h
=============

**/


enum {
   T_UNDEF,
   T_DISC
}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cstring>

struct torchtype
{
    static constexpr const char* T_UNDEF_ = "undef";
    static constexpr const char* T_DISC_  = "disc" ;

    static unsigned Type(const char* name); 
    static const char* Name(unsigned type); 
}; 

unsigned torchtype::Type(const char* name) 
{
    unsigned type = T_UNDEF ;
    if(strcmp(name,T_DISC_)==0) type = T_DISC ; 
    return type ; 
}

const char* torchtype::Name(unsigned type)
{
    const char* n = T_UNDEF_ ; 
    switch(type)
    {
        case T_UNDEF: n = T_UNDEF_ ; break ; 
        case T_DISC:  n = T_DISC_ ; break ; 
    }
    return n ; 
}
#endif


