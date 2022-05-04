#pragma once
/**
SBAS.h : Basis mode used by SGLM 
====================================
**/


enum { BAS_MANUAL, BAS_EXTENT, BAS_GAZELENGTH, BAS_NEARABS } ;  


#ifndef __CUDACC__
#include <cstring>
#include <cstdlib>
#include <cassert>

struct SBAS
{
    static const char* Name(int cam);
    static int        Type(const char* name);
    static constexpr const char* MANUAL_ = "manual" ;
    static constexpr const char* EXTENT_ = "extent" ;
    static constexpr const char* GAZELENGTH_ = "gazelength" ;
    static constexpr const char* NEARABS_ = "nearabs" ;
    static int EGet(const char* ekey, const char* fallback);   
};


inline int SBAS::EGet(const char* ekey, const char* fallback)
{
    const char* name_ = getenv(ekey) ; 
    const char* name = name_ ? name_ : fallback ; 
    assert( name ); 
    int bas = SBAS::Type(name ); 
    const char* name2 = SBAS::Name(bas) ; 
    bool consistent = strcmp( name, name2) == 0 ; 
    if(!consistent) printf("SBAS::EGet ERROR unknown name [%s]\n", name ) ;  
    assert(consistent ); 
    return bas ; 
}

inline const char* SBAS::Name(int cam )
{
    const char* s = nullptr ;
    switch(cam)
    {
        case BAS_MANUAL:     s = MANUAL_     ; break ;
        case BAS_EXTENT:     s = EXTENT_     ; break ;
        case BAS_GAZELENGTH: s = GAZELENGTH_ ; break ;
        case BAS_NEARABS:    s = NEARABS_ ; break ;
    }
    return s ; 
}
inline int SBAS::Type(const char* name)
{
    int type = BAS_MANUAL ;
    if(strcmp(name,MANUAL_) == 0 )     type = BAS_MANUAL ;
    if(strcmp(name,EXTENT_) == 0 )     type = BAS_EXTENT ;
    if(strcmp(name,GAZELENGTH_) == 0 ) type = BAS_GAZELENGTH ;
    if(strcmp(name,NEARABS_) == 0 )    type = BAS_NEARABS ;
    return type ;
}
#endif

