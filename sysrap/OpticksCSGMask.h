#pragma once


#include "OpticksCSG.h"

typedef enum {

   CSGMASK_UNION        = 0x1 << CSG_UNION , 
   CSGMASK_INTERSECTION = 0x1 << CSG_INTERSECTION ,
   CSGMASK_DIFFERENCE   = 0x1 << CSG_DIFFERENCE,
   CSGMASK_CYLINDER     = 0x1 << CSG_CYLINDER, 
   CSGMASK_DISC         = 0x1 << CSG_DISC, 
   CSGMASK_CONE         = 0x1 << CSG_CONE,
   CSGMASK_ZSPHERE      = 0x1 << CSG_ZSPHERE

} OpticksCSGMask_t ; 


