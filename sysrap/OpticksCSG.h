#pragma once

typedef enum {
    CSG_ZERO=0,
    CSG_UNION=1,
    CSG_INTERSECTION=2,
    CSG_DIFFERENCE=3,
    CSG_PARTLIST=4,   

    CSG_SPHERE=5,
       CSG_BOX=6,
   CSG_ZSPHERE=7,
     CSG_ZLENS=8,
       CSG_PMT=9,
     CSG_PRISM=10,
      CSG_TUBS=11,
 CSG_UNDEFINED=12

} OpticksCSG_t ; 
   

#ifndef __CUDACC__

#include <cstring>

static const char* CSG_ZERO_          = "zero" ; 
static const char* CSG_INTERSECTION_  = "intersection" ; 
static const char* CSG_UNION_         = "union" ; 
static const char* CSG_DIFFERENCE_    = "difference" ; 
static const char* CSG_PARTLIST_      = "partlist" ; 
static const char* CSG_SPHERE_        = "sphere" ; 
static const char* CSG_BOX_           = "box" ; 
static const char* CSG_ZSPHERE_       = "zsphere" ; 
static const char* CSG_ZLENS_         = "zlens" ; 
static const char* CSG_PMT_           = "pmt" ; 
static const char* CSG_PRISM_         = "prism" ; 
static const char* CSG_TUBS_          = "tubs" ; 
static const char* CSG_UNDEFINED_     = "undefined" ; 


static bool CSGIsPrimitive(OpticksCSG_t type)
{
    return !(type == CSG_INTERSECTION || type == CSG_UNION || type == CSG_DIFFERENCE) ; 
}

static char CSGChar(const char* nodename)
{
    char sc = 'U' ;
    if(     strcmp(nodename, CSG_BOX_) == 0)            sc = 'B' ;
    else if(strcmp(nodename, CSG_SPHERE_) == 0)         sc = 'S' ;
    else if(strcmp(nodename, CSG_ZSPHERE_) == 0)        sc = 'Z' ;
    else if(strcmp(nodename, CSG_ZLENS_) == 0)          sc = 'L' ;
    else if(strcmp(nodename, CSG_PMT_) == 0)            sc = 'P' ;  // not operational
    else if(strcmp(nodename, CSG_PRISM_) == 0)          sc = 'M' ;
    else if(strcmp(nodename, CSG_TUBS_) == 0)           sc = 'T' ;
    else if(strcmp(nodename, CSG_INTERSECTION_) == 0)   sc = 'I' ;
    else if(strcmp(nodename, CSG_UNION_) == 0)          sc = 'J' ;
    else if(strcmp(nodename, CSG_DIFFERENCE_) == 0)     sc = 'K' ;
    else if(strcmp(nodename, CSG_PARTLIST_) == 0)       sc = 'C' ;
    return sc ;
}


static OpticksCSG_t CSGTypeCode(const char* nodename)
{
    OpticksCSG_t tc = CSG_UNDEFINED ;
    if(     strcmp(nodename, CSG_BOX_) == 0)            tc = CSG_BOX ;
    else if(strcmp(nodename, CSG_SPHERE_) == 0)         tc = CSG_SPHERE ;
    else if(strcmp(nodename, CSG_ZSPHERE_) == 0)        tc = CSG_ZSPHERE ;
    else if(strcmp(nodename, CSG_ZLENS_) == 0)          tc = CSG_ZLENS ;
    else if(strcmp(nodename, CSG_PMT_) == 0)            tc = CSG_PMT ;  // not operational
    else if(strcmp(nodename, CSG_PRISM_) == 0)          tc = CSG_PRISM ;
    else if(strcmp(nodename, CSG_TUBS_) == 0)           tc = CSG_TUBS ;
    else if(strcmp(nodename, CSG_INTERSECTION_) == 0)   tc = CSG_INTERSECTION ;
    else if(strcmp(nodename, CSG_UNION_) == 0)          tc = CSG_UNION ;
    else if(strcmp(nodename, CSG_DIFFERENCE_) == 0)     tc = CSG_DIFFERENCE ;
    else if(strcmp(nodename, CSG_PARTLIST_) == 0)       tc = CSG_PARTLIST ;
    return tc ;
}



static OpticksCSG_t CSGFlag(char code)
{
    switch(code) 
    {   
       case 'B':return CSG_BOX     ; break ;
       case 'S':return CSG_SPHERE  ; break ;
       case 'Z':return CSG_ZSPHERE ; break ;
       case 'L':return CSG_ZLENS   ; break ;
       case 'P':return CSG_PMT     ; break ;
       case 'M':return CSG_PRISM     ; break ;
       case 'T':return CSG_TUBS     ; break ;
       case 'I':return CSG_INTERSECTION ; break ;
       case 'J':return CSG_UNION        ; break ;
       case 'K':return CSG_DIFFERENCE   ; break ;
       case 'C':return CSG_PARTLIST   ; break ;
       case 'U':return CSG_UNDEFINED ; break ;
    }   
    return CSG_ZERO ;
} 



static const char* CSGName( OpticksCSG_t csg )
{
    const char* s = NULL ; 
    switch(csg)
    {
        case CSG_ZERO:          s = CSG_ZERO_          ; break ; 
        case CSG_INTERSECTION:  s = CSG_INTERSECTION_  ; break ; 
        case CSG_UNION:         s = CSG_UNION_         ; break ; 
        case CSG_DIFFERENCE:    s = CSG_DIFFERENCE_    ; break ; 
        case CSG_PARTLIST:      s = CSG_PARTLIST_      ; break ; 
        case CSG_SPHERE:        s = CSG_SPHERE_        ; break ; 
        case CSG_BOX:           s = CSG_BOX_           ; break ; 
        case CSG_ZSPHERE:       s = CSG_ZSPHERE_       ; break ; 
        case CSG_ZLENS:         s = CSG_ZLENS_         ; break ; 
        case CSG_PMT:           s = CSG_PMT_           ; break ; 
        case CSG_PRISM:         s = CSG_PRISM_         ; break ; 
        case CSG_TUBS:          s = CSG_TUBS_          ; break ; 
        case CSG_UNDEFINED:     s = CSG_UNDEFINED_     ; break ; 
    }
    return s ; 
}

static const char* CSGChar2Name(char code )
{
    OpticksCSG_t flag = CSGFlag(code) ; 
    return CSGName(flag) ;
}


#endif

