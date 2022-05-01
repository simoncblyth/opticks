#pragma once
/**
SRG.h : Raygen Mode
=====================

**/


enum { SRG_RENDER, SRG_SIMTRACE, SRG_SIMULATE } ;  


#ifndef __CUDACC__
#include <cstdint>
#include <cstring>
struct SRG 
{
    static const char* Name(int32_t raygenmode);
    static int32_t     Type(const char* name);
    static constexpr const char* RENDER_ = "render" ;
    static constexpr const char* SIMTRACE_ = "simtrace" ;
    static constexpr const char* SIMULATE_ = "simulate" ;
};
inline const char* SRG::Name(int32_t raygenmode)
{
    const char* s = nullptr ;
    switch(raygenmode)
    {
        case SRG_RENDER:   s = RENDER_   ; break ;
        case SRG_SIMTRACE: s = SIMTRACE_ ; break ;
        case SRG_SIMULATE: s = SIMULATE_ ; break ;
    }
    return s ; 
}
inline int32_t SRG::Type(const char* name)
{
    int32_t type = SRG_RENDER ;
    if(strcmp(name,RENDER_) == 0 )   type = SRG_RENDER ;
    if(strcmp(name,SIMTRACE_) == 0 ) type = SRG_SIMTRACE ;
    if(strcmp(name,SIMULATE_) == 0 ) type = SRG_SIMULATE ;
    return type ;
}
#endif

