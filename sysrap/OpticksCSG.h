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
  CSG_CYLINDER=12,
 CSG_UNDEFINED=13,

 CSG_FLAGPARTLIST=100,
 CSG_FLAGNODETREE=101

} OpticksCSG_t ; 
   

/*
* keep CSG_SPHERE as 1st primitive
* keep CSG_UNDEFINED as one beyond the last primitive
*/

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
static const char* CSG_CYLINDER_      = "cylinder" ; 
static const char* CSG_UNDEFINED_     = "undefined" ; 

static const char* CSG_FLAGPARTLIST_ = "flagpartlist" ; 
static const char* CSG_FLAGNODETREE_ = "flagnodetree" ; 



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
    else if(strcmp(nodename, CSG_CYLINDER_) == 0)       tc = CSG_CYLINDER ;
    else if(strcmp(nodename, CSG_INTERSECTION_) == 0)   tc = CSG_INTERSECTION ;
    else if(strcmp(nodename, CSG_UNION_) == 0)          tc = CSG_UNION ;
    else if(strcmp(nodename, CSG_DIFFERENCE_) == 0)     tc = CSG_DIFFERENCE ;
    else if(strcmp(nodename, CSG_PARTLIST_) == 0)       tc = CSG_PARTLIST ;
    else if(strcmp(nodename, CSG_FLAGPARTLIST_) == 0)   tc = CSG_FLAGPARTLIST ;
    else if(strcmp(nodename, CSG_FLAGNODETREE_) == 0)   tc = CSG_FLAGNODETREE ;
    return tc ;
}


static const char* CSGName( OpticksCSG_t type )
{
    const char* s = NULL ; 
    switch(type)
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
        case CSG_CYLINDER:      s = CSG_CYLINDER_      ; break ; 
        case CSG_UNDEFINED:     s = CSG_UNDEFINED_     ; break ; 
        case CSG_FLAGPARTLIST:  s = CSG_FLAGPARTLIST_  ; break ; 
        case CSG_FLAGNODETREE:  s = CSG_FLAGNODETREE_  ; break ; 
    }
    return s ; 
}

static bool CSGExists( OpticksCSG_t type )
{ 
   return CSGName(type) != NULL ;
}

static bool CSGIsPrimitive(OpticksCSG_t type)
{
    return !(type == CSG_INTERSECTION || type == CSG_UNION || type == CSG_DIFFERENCE) ; 
}



#endif

