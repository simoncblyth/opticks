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
      CSG_SLAB=13,
     CSG_PLANE=14,

      CSG_CONE=15,
 CSG_MULTICONE=16,
      CSG_BOX3=17,
 CSG_TRAPEZOID=18,
 CSG_CONVEXPOLYHEDRON=19,
     CSG_DISC=20,
   CSG_SEGMENT=21,
   CSG_ELLIPSOID=22,
   CSG_TORUS=23,
   CSG_HYPERBOLOID=24,
   CSG_CUBIC=25,
 CSG_UNDEFINED=26,

 CSG_FLAGPARTLIST=100,
 CSG_FLAGNODETREE=101,
 CSG_FLAGINVISIBLE=102

} OpticksCSG_t ; 


   

/*
* keep CSG_SPHERE as 1st primitive
* keep CSG_UNDEFINED as one beyond the last primitive
* regenerate the derived python e-n-u-m-e-r-a-t-i-o-n classes with sysrap-csg-generate  
* CAUTION : the text of this header is parsed for the generation, 
  some words are not allowed and avoid semi-colons


TODO: stick these in a struct 


*/

#ifndef __CUDACC__

#include <string>
#include <cstring>
#include <cassert>

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
static const char* CSG_DISC_          = "disc" ; 
static const char* CSG_SLAB_          = "slab" ; 
static const char* CSG_PLANE_         = "plane" ; 
static const char* CSG_CONE_          = "cone" ; 
static const char* CSG_MULTICONE_     = "multicone" ; 
static const char* CSG_BOX3_          = "box3" ; 
static const char* CSG_TRAPEZOID_     = "trapezoid" ; 
static const char* CSG_CONVEXPOLYHEDRON_ = "convexpolyhedron" ; 
static const char* CSG_SEGMENT_       = "segment" ; 
static const char* CSG_ELLIPSOID_       = "ellipsoid" ; 
static const char* CSG_TORUS_          = "torus" ; 
static const char* CSG_HYPERBOLOID_    = "hyperboloid" ; 
static const char* CSG_CUBIC_          = "cubic" ; 
static const char* CSG_UNDEFINED_     = "undefined" ; 

static const char* CSG_FLAGPARTLIST_ = "flagpartlist" ; 
static const char* CSG_FLAGNODETREE_ = "flagnodetree" ; 
static const char* CSG_FLAGINVISIBLE_ = "flaginvisible" ; 



struct CSG
{
    static OpticksCSG_t TypeCode(const char* nodename)
    {
        OpticksCSG_t tc = CSG_UNDEFINED ;
        if(     strcmp(nodename, CSG_BOX_) == 0)            tc = CSG_BOX ;
        else if(strcmp(nodename, CSG_BOX3_) == 0)           tc = CSG_BOX3 ;
        else if(strcmp(nodename, CSG_SPHERE_) == 0)         tc = CSG_SPHERE ;
        else if(strcmp(nodename, CSG_ZSPHERE_) == 0)        tc = CSG_ZSPHERE ;
        else if(strcmp(nodename, CSG_ZLENS_) == 0)          tc = CSG_ZLENS ;
        else if(strcmp(nodename, CSG_PMT_) == 0)            tc = CSG_PMT ;  // not operational
        else if(strcmp(nodename, CSG_PRISM_) == 0)          tc = CSG_PRISM ;
        else if(strcmp(nodename, CSG_TUBS_) == 0)           tc = CSG_TUBS ;
        else if(strcmp(nodename, CSG_CYLINDER_) == 0)       tc = CSG_CYLINDER ;
        else if(strcmp(nodename, CSG_DISC_) == 0)           tc = CSG_DISC ;
        else if(strcmp(nodename, CSG_SLAB_) == 0)           tc = CSG_SLAB ;
        else if(strcmp(nodename, CSG_PLANE_) == 0)          tc = CSG_PLANE ;
        else if(strcmp(nodename, CSG_CONE_) == 0)           tc = CSG_CONE ;
        else if(strcmp(nodename, CSG_MULTICONE_) == 0)      tc = CSG_MULTICONE ;
        else if(strcmp(nodename, CSG_TRAPEZOID_) == 0)      tc = CSG_TRAPEZOID ;
        else if(strcmp(nodename, CSG_ELLIPSOID_) == 0)      tc = CSG_ELLIPSOID ;
        else if(strcmp(nodename, CSG_TORUS_) == 0)          tc = CSG_TORUS ;
        else if(strcmp(nodename, CSG_HYPERBOLOID_) == 0)    tc = CSG_HYPERBOLOID ;
        else if(strcmp(nodename, CSG_CUBIC_) == 0)          tc = CSG_CUBIC ;
        else if(strcmp(nodename, CSG_SEGMENT_) == 0)        tc = CSG_SEGMENT ;
        else if(strcmp(nodename, CSG_CONVEXPOLYHEDRON_) == 0) tc = CSG_CONVEXPOLYHEDRON ;
        else if(strcmp(nodename, CSG_INTERSECTION_) == 0)   tc = CSG_INTERSECTION ;
        else if(strcmp(nodename, CSG_UNION_) == 0)          tc = CSG_UNION ;
        else if(strcmp(nodename, CSG_DIFFERENCE_) == 0)     tc = CSG_DIFFERENCE ;
        else if(strcmp(nodename, CSG_PARTLIST_) == 0)       tc = CSG_PARTLIST ;
        else if(strcmp(nodename, CSG_FLAGPARTLIST_) == 0)   tc = CSG_FLAGPARTLIST ;
        else if(strcmp(nodename, CSG_FLAGNODETREE_) == 0)   tc = CSG_FLAGNODETREE ;
        else if(strcmp(nodename, CSG_FLAGINVISIBLE_) == 0)  tc = CSG_FLAGINVISIBLE ;
        return tc ;
    }

    static OpticksCSG_t DeMorganSwap( OpticksCSG_t type )
    {
        OpticksCSG_t t = CSG_ZERO ; 
        switch(type)
        {
            case CSG_INTERSECTION:  t = CSG_UNION         ; break ; 
            case CSG_UNION:         t = CSG_INTERSECTION  ; break ; 
            default:                t = CSG_ZERO          ; break ; 
        }
        assert( t != CSG_ZERO ); 
        return t ; 
    }

    static const char* Name( OpticksCSG_t type )
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
            case CSG_BOX3:          s = CSG_BOX3_          ; break ; 
            case CSG_ZSPHERE:       s = CSG_ZSPHERE_       ; break ; 
            case CSG_ZLENS:         s = CSG_ZLENS_         ; break ; 
            case CSG_PMT:           s = CSG_PMT_           ; break ; 
            case CSG_PRISM:         s = CSG_PRISM_         ; break ; 
            case CSG_TUBS:          s = CSG_TUBS_          ; break ; 
            case CSG_CYLINDER:      s = CSG_CYLINDER_      ; break ; 
            case CSG_DISC:          s = CSG_DISC_          ; break ; 
            case CSG_SLAB:          s = CSG_SLAB_          ; break ; 
            case CSG_PLANE:         s = CSG_PLANE_         ; break ; 
            case CSG_CONE:          s = CSG_CONE_          ; break ; 
            case CSG_MULTICONE:     s = CSG_MULTICONE_     ; break ; 
            case CSG_TRAPEZOID:     s = CSG_TRAPEZOID_     ; break ; 
            case CSG_ELLIPSOID:     s = CSG_ELLIPSOID_     ; break ; 
            case CSG_TORUS:         s = CSG_TORUS_         ; break ; 
            case CSG_HYPERBOLOID:   s = CSG_HYPERBOLOID_   ; break ; 
            case CSG_CUBIC:         s = CSG_CUBIC_         ; break ; 
            case CSG_SEGMENT:       s = CSG_SEGMENT_       ; break ; 
            case CSG_CONVEXPOLYHEDRON: s = CSG_CONVEXPOLYHEDRON_ ; break ; 
            case CSG_UNDEFINED:     s = CSG_UNDEFINED_     ; break ; 
            case CSG_FLAGPARTLIST:  s = CSG_FLAGPARTLIST_  ; break ; 
            case CSG_FLAGNODETREE:  s = CSG_FLAGNODETREE_  ; break ; 
            case CSG_FLAGINVISIBLE: s = CSG_FLAGINVISIBLE_ ; break ; 
        }
        return s ; 
    }

    static std::string Tag( OpticksCSG_t type )
    {
        const char* name = Name(type);
        assert(strlen(name) > 2 );
        std::string s(name, name+2) ; 
        return s ; 
    }

    static bool Exists( OpticksCSG_t type )
    { 
       return Name(type) != NULL ;
    }

    static bool IsPrimitive(OpticksCSG_t type)
    {
        return !(type == CSG_INTERSECTION || type == CSG_UNION || type == CSG_DIFFERENCE) ; 
    }

    static bool HasPlanes(OpticksCSG_t type)
    {
        return (type == CSG_TRAPEZOID || type == CSG_CONVEXPOLYHEDRON || type == CSG_SEGMENT ) ; 
    }
};


#endif

