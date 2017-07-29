#pragma once


#include <string>
#include <sstream>


typedef enum 
{
    SOLVE_VECGEOM      = 0x1 << 0,
    SOLVE_UNOBFUSCATED = 0x1 << 1,
    SOLVE_ROBUSTQUAD   = 0x1 << 2


} SolveType_t ; 

static const char* SOLVE_VECGEOM_      = "VECGEOM" ; 
static const char* SOLVE_UNOBFUSCATED_ = "UNOBFUSCATED" ; 
static const char* SOLVE_ROBUSTQUAD_   = "ROBUSTQUAD" ; 



static const char* SolveType(SolveType_t type)
{
    const char* s = NULL ; 
    switch(type)
    {
        case SOLVE_VECGEOM     : s = SOLVE_VECGEOM_      ; break ; 
        case SOLVE_UNOBFUSCATED: s = SOLVE_UNOBFUSCATED_ ; break ; 
        case SOLVE_ROBUSTQUAD:   s = SOLVE_ROBUSTQUAD_   ; break ; 
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


