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
    CSG_UNDEFINED=11
} OpticksCSG_t ; 
   

#ifndef __CUDACC__

static const char* CSG_ZERO_          = "ZERO" ; 
static const char* CSG_INTERSECTION_  = "INTERSECTION" ; 
static const char* CSG_UNION_         = "UNION" ; 
static const char* CSG_DIFFERENCE_    = "DIFFERENCE" ; 
static const char* CSG_PARTLIST_      = "PARTLIST" ; 
static const char* CSG_SPHERE_        = "SPHERE" ; 
static const char* CSG_BOX_           = "BOX" ; 
static const char* CSG_ZSPHERE_       = "ZSPHERE" ; 
static const char* CSG_ZLENS_         = "ZLENS" ; 
static const char* CSG_PMT_           = "PMT" ; 
static const char* CSG_PRISM_         = "PRISM" ; 
static const char* CSG_UNDEFINED_     = "UNDEFINED" ; 

OpticksCSG_t CSGFlag(char nodecode)
{
    switch(nodecode) 
    {   
       case 'B':return CSG_BOX     ; break ;
       case 'S':return CSG_SPHERE  ; break ;
       case 'Z':return CSG_ZSPHERE ; break ;
       case 'L':return CSG_ZLENS   ; break ;
       case 'P':return CSG_PMT     ; break ;
       case 'M':return CSG_PRISM     ; break ;
       case 'I':return CSG_INTERSECTION ; break ;
       case 'J':return CSG_UNION        ; break ;
       case 'K':return CSG_DIFFERENCE   ; break ;
       case 'U':return CSG_UNDEFINED ; break ;
    }   
    return CSG_ZERO ;
} 

char CSGChar(const char* nodename)
{
    char sc = 'U' ;
    if(     strcmp(nodename, CSG_BOX_) == 0)            sc = 'B' ;
    else if(strcmp(nodename, CSG_SPHERE_) == 0)         sc = 'S' ;
    else if(strcmp(nodename, CSG_ZSPHERE_) == 0)        sc = 'Z' ;
    else if(strcmp(nodename, CSG_ZLENS_) == 0)          sc = 'L' ;
    else if(strcmp(nodename, CSG_PMT_) == 0)            sc = 'P' ;  // not operational
    else if(strcmp(nodename, CSG_PRISM_) == 0)          sc = 'M' ;
    else if(strcmp(nodename, CSG_INTERSECTION_) == 0)   sc = 'I' ;
    else if(strcmp(nodename, CSG_UNION_) == 0)          sc = 'J' ;
    else if(strcmp(nodename, CSG_DIFFERENCE_) == 0)     sc = 'K' ;
    return sc ;
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
        case CSG_UNDEFINED:     s = CSG_UNDEFINED_     ; break ; 
    }
    return s ; 
}

#endif

