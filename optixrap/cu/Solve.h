#pragma once


#include <string>
#include <sstream>


typedef enum 
{
    SOLVE_VECGEOM        = 0x1 << 0,
    SOLVE_UNOBFUSCATED   = 0x1 << 1,
    SOLVE_ROBUST_VIETA   = 0x1 << 2,
    SOLVE_ROBUSTQUAD_1   = 0x1 << 3,
    SOLVE_ROBUSTCUBIC_0  = 0x1 << 4,
    SOLVE_ROBUSTCUBIC_1  = 0x1 << 5,
    SOLVE_ROBUSTCUBIC_2  = 0x1 << 6

} SolveType_t ; 

static const char* SOLVE_VECGEOM_      = "VECGEOM" ; 
static const char* SOLVE_UNOBFUSCATED_ = "UNOBFUSCATED" ; 
static const char* SOLVE_ROBUST_VIETA_   = "ROBUST_VIETA" ; 
static const char* SOLVE_ROBUSTQUAD_1_   = "ROBUSTQUAD_1" ; 
static const char* SOLVE_ROBUSTCUBIC_0_  = "ROBUSTCUBIC_0" ; 
static const char* SOLVE_ROBUSTCUBIC_1_  = "ROBUSTCUBIC_1" ; 
static const char* SOLVE_ROBUSTCUBIC_2_  = "ROBUSTCUBIC_2" ; 


static const char* SolveType(SolveType_t type)
{
    const char* s = NULL ; 
    switch(type)
    {
        case SOLVE_VECGEOM      : s = SOLVE_VECGEOM_       ; break ; 
        case SOLVE_UNOBFUSCATED : s = SOLVE_UNOBFUSCATED_  ; break ; 
        case SOLVE_ROBUST_VIETA : s = SOLVE_ROBUST_VIETA_  ; break ; 
        case SOLVE_ROBUSTQUAD_1 : s = SOLVE_ROBUSTQUAD_1_  ; break ; 
        case SOLVE_ROBUSTCUBIC_0: s = SOLVE_ROBUSTCUBIC_0_ ; break ; 
        case SOLVE_ROBUSTCUBIC_1: s = SOLVE_ROBUSTCUBIC_1_ ; break ; 
        case SOLVE_ROBUSTCUBIC_2: s = SOLVE_ROBUSTCUBIC_2_ ; break ; 
    }  
    return s ; 
}


static std::string SolveTypeMask(unsigned mask)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < 32 ; i++) 
    {
        SolveType_t typ = (SolveType_t)(0x1 << i) ;  
        if(mask & typ) ss << SolveType(typ) << " " ; 
    }
    return ss.str();
}


