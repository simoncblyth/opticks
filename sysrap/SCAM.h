#pragma once
/**
SCAM.h :
=====================
**/


enum { CAM_PERSPECTIVE, CAM_ORTHOGRAPHIC, CAM_EQUIRECTANGULAR } ;  


#ifndef __CUDACC__
#include <cstring>
#include <cstdlib>
#include <cassert>

struct SCAM
{
    static const char* Name(int cam);
    static int        Type(const char* name);
    static constexpr const char* PERSPECTIVE_ = "perspective" ;
    static constexpr const char* ORTHOGRAPHIC_ = "orthographic" ;
    static constexpr const char* EQUIRECTANGULAR_ = "equirectangular" ;
    static int EGet(const char* ekey, const char* fallback);   
};


inline int SCAM::EGet(const char* ekey, const char* fallback)
{
    const char* name_ = getenv(ekey) ; 
    const char* name = name_ ? name_ : fallback ; 
    assert( name ); 
    int cam = SCAM::Type(name ); 
    const char* name2 = SCAM::Name(cam) ; 
    bool consistent = strcmp( name, name2) == 0 ; 
    if(!consistent) printf("SCAM::EGet ERROR unknown name [%s]\n", name ) ;  
    assert(consistent ); 
    return cam ; 
}

inline const char* SCAM::Name(int cam )
{
    const char* s = nullptr ;
    switch(cam)
    {
        case CAM_PERSPECTIVE:     s = PERSPECTIVE_     ; break ;
        case CAM_ORTHOGRAPHIC:    s = ORTHOGRAPHIC_    ; break ;
        case CAM_EQUIRECTANGULAR: s = EQUIRECTANGULAR_ ; break ;
    }
    return s ; 
}
inline int SCAM::Type(const char* name)
{
    int type = CAM_PERSPECTIVE ;
    if(strcmp(name,PERSPECTIVE_) == 0 )     type = CAM_PERSPECTIVE ;
    if(strcmp(name,ORTHOGRAPHIC_) == 0 )    type = CAM_ORTHOGRAPHIC ;
    if(strcmp(name,EQUIRECTANGULAR_) == 0 ) type = CAM_EQUIRECTANGULAR ;
    return type ;
}
#endif

