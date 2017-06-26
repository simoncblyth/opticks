#pragma once


#include "OpticksCSG.h"

typedef enum {

   CSGMASK_UNION    = 0x1 << CSG_UNION , 
   CSGMASK_CYLINDER = 0x1 << CSG_CYLINDER, 
   CSGMASK_CONE     = 0x1 << CSG_CONE

} OpticksCSGMask_t ; 


