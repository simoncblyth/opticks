#pragma once

enum { RG_RENDER, RG_SIMTRACE, RG_SIMULATE } ;  


#ifndef __CUDACC__
#include <cstring>
struct RG 
{
    static const char* Name(int32_t raygenmode);
    static int32_t     Type(const char* name);
    static constexpr const char* RENDER_ = "render" ;
    static constexpr const char* SIMTRACE_ = "simtrace" ;
    static constexpr const char* SIMULATE_ = "simulate" ;
};
inline const char* RG::Name(int32_t raygenmode)
{
    const char* s = nullptr ;
    switch(raygenmode)
    {
        case RG_RENDER:   s = RENDER_   ; break ;
        case RG_SIMTRACE: s = SIMTRACE_ ; break ;
        case RG_SIMULATE: s = SIMULATE_ ; break ;
    }
    return s ; 
}
inline int32_t RG::Type(const char* name)
{
    int32_t type = RG_RENDER ;
    if(strcmp(name,RENDER_) == 0 )   type = RG_RENDER ;
    if(strcmp(name,SIMTRACE_) == 0 ) type = RG_SIMTRACE ;
    if(strcmp(name,SIMULATE_) == 0 ) type = RG_SIMULATE ;
    return type ;
}
#endif

