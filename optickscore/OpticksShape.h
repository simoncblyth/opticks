#pragma once

typedef enum {
    SHAPE_UNDEFINED     = 0x1 << 0 , 
    SHAPE_INTERSECTION  = 0x1 << 1 , 
    SHAPE_UNION         = 0x1 << 2 , 
    SHAPE_DIFFERENCE    = 0x1 << 3 ,
    SHAPE_PRIMITIVE     = 0x1 << 4 , 
    SHAPE_COMPOSITE     = 0x1 << 5 ,
    SHAPE_CONSTITUENT   = 0x1 << 6 ,
    SHAPE_CONSTITUENT_A = 0x1 << 7 ,
    SHAPE_CONSTITUENT_B = 0x1 << 8 ,
    SHAPE_BOOLEAN       = 0x1 << 9 
} OpticksShape_t ; 



#ifndef __CUDACC__

static const char* SHAPE_UNDEFINED_     = "SHAPE_UNDEFINED" ; 
static const char* SHAPE_INTERSECTION_  = "SHAPE_INTERSECTION" ; 
static const char* SHAPE_UNION_         = "SHAPE_UNION" ; 
static const char* SHAPE_DIFFERENCE_    = "SHAPE_DIFFERENCE" ; 
static const char* SHAPE_PRIMITIVE_     = "SHAPE_PRIMITIVE" ; 
static const char* SHAPE_COMPOSITE_     = "SHAPE_COMPOSITE" ; 
static const char* SHAPE_CONSTITUENT_   = "SHAPE_CONSTITUENT" ; 
static const char* SHAPE_CONSTITUENT_A_ = "SHAPE_CONSTITUENT_A" ; 
static const char* SHAPE_CONSTITUENT_B_ = "SHAPE_CONSTITUENT_B" ; 
static const char* SHAPE_BOOLEAN_       = "SHAPE_BOOLEAN" ; 


static const char* ShapeName( OpticksShape_t flag )
{
    const char* s = NULL ; 
    switch(flag)
    {
        case SHAPE_UNDEFINED:     s = SHAPE_UNDEFINED_     ; break ; 
        case SHAPE_INTERSECTION:  s = SHAPE_INTERSECTION_  ; break ; 
        case SHAPE_UNION:         s = SHAPE_UNION_         ; break ; 
        case SHAPE_DIFFERENCE:    s = SHAPE_DIFFERENCE_    ; break ; 
        case SHAPE_PRIMITIVE:     s = SHAPE_PRIMITIVE_     ; break ; 
        case SHAPE_COMPOSITE:     s = SHAPE_COMPOSITE_     ; break ; 
        case SHAPE_CONSTITUENT:   s = SHAPE_CONSTITUENT_   ; break ; 
        case SHAPE_CONSTITUENT_A: s = SHAPE_CONSTITUENT_A_ ; break ; 
        case SHAPE_CONSTITUENT_B: s = SHAPE_CONSTITUENT_B_ ; break ; 
        case SHAPE_BOOLEAN:       s = SHAPE_BOOLEAN_       ; break ; 
    }
    return s ; 
}

#include <string>
#include <sstream>

static std::string ShapeMask( unsigned flag )
{
    std::stringstream ss ; 
    if(flag & SHAPE_UNDEFINED)     ss << SHAPE_UNDEFINED_    << " " ; 
    if(flag & SHAPE_INTERSECTION)  ss << SHAPE_INTERSECTION_ << " " ; 
    if(flag & SHAPE_UNION)         ss << SHAPE_UNION_ << " " ; 
    if(flag & SHAPE_DIFFERENCE)    ss << SHAPE_DIFFERENCE_ << " " ; 
    if(flag & SHAPE_PRIMITIVE)     ss << SHAPE_PRIMITIVE_ << " " ; 
    if(flag & SHAPE_COMPOSITE)     ss << SHAPE_COMPOSITE_ << " " ; 
    if(flag & SHAPE_CONSTITUENT)   ss << SHAPE_CONSTITUENT_ << " " ; 
    if(flag & SHAPE_CONSTITUENT_A) ss << SHAPE_CONSTITUENT_A_ << " " ; 
    if(flag & SHAPE_CONSTITUENT_B) ss << SHAPE_CONSTITUENT_B_ << " " ; 
    if(flag & SHAPE_BOOLEAN)       ss << SHAPE_BOOLEAN_ << " " ; 
    return ss.str();
}

#endif

