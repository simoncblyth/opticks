#pragma once

typedef enum {
      ZERO, 
      SPHERE, 
      TUBS, 
      BOX,
      PRISM
      } NPart_t ;  

/*
  Boolean ops now handled in OpticksShape.h 

      INTERSECTION,
      UNION,
      DIFFERENCE
*/


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
