#pragma once

typedef enum {
      EMPTY=-1, 
      ZERO=0, 
      SPHERE=1, 
      TUBS=2, 
      BOX=3,
      PRISM=4
      } NPart_t ;  

/*
  Boolean ops now handled in OpticksShape.h 

      INTERSECTION,
      UNION,
      DIFFERENCE
*/


// these internals exposed, 
// as still being used at higher level in ggeo-/GParts
// moved here from .hpp as "needed"  GPU side
// quotes as the q* are used

enum { 
  PARAM_J  = 0, 
  PARAM_K  = 0 };       // q0.f.xyzw


// (1,0) used for sizeZ in ZTubs // q1.u.x
enum { 
   INDEX_J    = 1, 
   INDEX_K    = 1 
};   // q1.u.y

enum { 
   BOUNDARY_J = 1, 
   BOUNDARY_K = 2 
};   // q1.u.z

enum { 
   FLAGS_J    = 1, 
   FLAGS_K    = 3 
};   // q1.u.w

enum { 
    BBMIN_J     = 2, 
    BBMIN_K     = 0 
};  // q2.f.xyz
enum { 
    TYPECODE_J  = 2, 
    TYPECODE_K  = 3 
};  // q2.u.w

enum { 
    BBMAX_J     = 3,     
    BBMAX_K = 0 
};  // q3.f.xyz 

enum { 
    NODEINDEX_J = 3, 
    NODEINDEX_K = 3 
};  // q3.u.w 





#ifndef __CUDACC__

static const char* PART_ZERO_   = "PART_ZERO" ; 
static const char* PART_SPHERE_ = "PART_SPHERE" ; 
static const char* PART_TUBS_   = "PART_TUBS" ; 
static const char* PART_BOX_    = "PART_BOX" ; 
static const char* PART_PRISM_  = "PART_PRISM" ; 

static const char* PartName( NPart_t partType )
{
    const char* s = 0 ; 
    switch(partType)
    {   
        case ZERO:    s = PART_ZERO_   ; break ; 
        case SPHERE:  s = PART_SPHERE_ ; break ; 
        case TUBS:    s = PART_TUBS_   ; break ; 
        case BOX:     s = PART_BOX_    ; break ; 
        case PRISM:   s = PART_PRISM_  ; break ; 
    }   
    return s ; 
}

#endif
