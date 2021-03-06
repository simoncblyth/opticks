/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once


#include "OpticksCSG.h"

typedef enum {

   CSGMASK_UNION        = 0x1 << CSG_UNION , 
   CSGMASK_INTERSECTION = 0x1 << CSG_INTERSECTION ,
   CSGMASK_DIFFERENCE   = 0x1 << CSG_DIFFERENCE,
   CSGMASK_CYLINDER     = 0x1 << CSG_CYLINDER, 
   CSGMASK_DISC         = 0x1 << CSG_DISC, 
   CSGMASK_CONE         = 0x1 << CSG_CONE,
   CSGMASK_ZSPHERE      = 0x1 << CSG_ZSPHERE,
   CSGMASK_BOX3         = 0x1 << CSG_BOX3

} OpticksCSGMask_t ; 


#ifndef __CUDACC__

#include <string>
#include <sstream>

static std::string CSGMaskDesc( unsigned mask )
{
    std::stringstream ss ;        
    if( mask & CSGMASK_UNION )        ss << "union " ; 
    if( mask & CSGMASK_INTERSECTION ) ss << "intersection " ; 
    if( mask & CSGMASK_DIFFERENCE )   ss << "difference " ; 
    return ss.str()  ; 
}

/**
CSG_MonoOperator
------------------

For masks corresponding to a single operator return
the CSG code of the operator, otherwise return CSG_ZERO.

TODO: collect these into a OpticksCSGMask class as static methods for easier identification 

**/

static OpticksCSG_t CSG_MonoOperator( unsigned mask )
{
    OpticksCSG_t op = CSG_ZERO ; 
    switch( mask )
    {
        case CSGMASK_UNION         : op = CSG_UNION          ; break ; 
        case CSGMASK_INTERSECTION  : op = CSG_INTERSECTION   ; break ;
        case CSGMASK_DIFFERENCE    : op = CSG_ZERO           ; break ;  // <-- difference is unusable for tree manipulation so give zero
        default                    : op = CSG_ZERO           ; break ; 
    } 
    return op ;
}

#endif

