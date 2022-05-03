#pragma once
/**
SCAM.h :
=====================
**/


enum { CAM_PERSPECTIVE, CAM_ORTHOGRAPHIC, CAM_EQUIRECTANGULAR } ;  


#ifndef __CUDACC__
#include <cstring>
#include <cstdlib>
struct SCAM
{
    static const char* Name(int cam);
    static int        Type(const char* name);
    static constexpr const char* PERSPECTIVE_ = "perspective" ;
    static constexpr const char* ORTHOGRAPHIC_ = "orthographic" ;
    static constexpr const char* EQUIRECTANGULAR_ = "equirectangular" ;
    static int EValue(const char* ekey, const char* fallback);   
};


inline int SCAM::EValue(const char* ekey, const char* fallback)
{
    const char* cam = getenv(ekey) ; 
    return SCAM::Type( cam ? cam : fallback ); 
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

