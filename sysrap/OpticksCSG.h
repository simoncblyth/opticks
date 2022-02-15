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

    CSG_TREE=1,
        CSG_UNION=2,
        CSG_INTERSECTION=3,
        CSG_DIFFERENCE=4,

    CSG_NODE=10, 
        CSG_CONTIGUOUS=11, 
        CSG_DISCONTIGUOUS=12,
        CSG_OVERLAP=13, 

    CSG_LEAF=100,
        CSG_SPHERE=101,
        CSG_BOX=102,
        CSG_ZSPHERE=103,
        CSG_TUBS=104,
        CSG_CYLINDER=105,
        CSG_SLAB=106,
        CSG_PLANE=107,
        CSG_CONE=108,
        CSG_MULTICONE=109,
        CSG_BOX3=110,
        CSG_TRAPEZOID=111,
        CSG_CONVEXPOLYHEDRON=112,
        CSG_DISC=113,
        CSG_SEGMENT=114,
        CSG_ELLIPSOID=115,
        CSG_TORUS=116,
        CSG_HYPERBOLOID=117,
        CSG_CUBIC=118,
        CSG_INFCYLINDER=119,
        CSG_PHICUT=120, 
        CSG_LPHICUT=121, 
        CSG_THETACUT=122, 
        CSG_LTHETACUT=123, 
        CSG_UNDEFINED=124, 

    CSG_OBSOLETE=1000, 
        CSG_PARTLIST=1001,   
        CSG_FLAGPARTLIST=1002,
        CSG_FLAGNODETREE=1003,
        CSG_FLAGINVISIBLE=1004,
        CSG_PMT=1005,
        CSG_ZLENS=1006,
        CSG_PRISM=1007,
        CSG_LAST=1008

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
#include <sstream>
#include <cstring>
#include <cassert>

static const char* CSG_ZERO_          = "zero" ; 


static const char* CSG_TREE_          = "tree" ; 
static const char* CSG_INTERSECTION_  = "intersection" ; 
static const char* CSG_UNION_         = "union" ; 
static const char* CSG_DIFFERENCE_    = "difference" ; 

static const char* CSG_NODE_           = "node" ; 
static const char* CSG_CONTIGUOUS_     = "contiguous" ; 
static const char* CSG_DISCONTIGUOUS_  = "discontiguous" ; 
static const char* CSG_OVERLAP_        = "overlap" ; 


static const char* CSG_LEAF_           = "leaf" ; 
static const char* CSG_SPHERE_        = "sphere" ; 
static const char* CSG_BOX_           = "box" ; 
static const char* CSG_ZSPHERE_       = "zsphere" ; 
static const char* CSG_TUBS_          = "tubs" ; 
static const char* CSG_CYLINDER_      = "cylinder" ; 
static const char* CSG_SLAB_          = "slab" ; 
static const char* CSG_PLANE_         = "plane" ; 
static const char* CSG_CONE_          = "cone" ; 
static const char* CSG_MULTICONE_     = "multicone" ; 
static const char* CSG_BOX3_          = "box3" ; 
static const char* CSG_TRAPEZOID_     = "trapezoid" ; 
static const char* CSG_CONVEXPOLYHEDRON_ = "convexpolyhedron" ; 
static const char* CSG_DISC_          = "disc" ; 
static const char* CSG_SEGMENT_       = "segment" ; 
static const char* CSG_ELLIPSOID_     = "ellipsoid" ; 
static const char* CSG_TORUS_          = "torus" ; 
static const char* CSG_HYPERBOLOID_    = "hyperboloid" ; 
static const char* CSG_CUBIC_          = "cubic" ; 
static const char* CSG_INFCYLINDER_   = "infcylinder" ; 
static const char* CSG_PHICUT_        = "phicut" ; 
static const char* CSG_LPHICUT_       = "lphicut" ; 
static const char* CSG_THETACUT_      = "thetacut" ; 
static const char* CSG_LTHETACUT_     = "lthetacut" ; 
static const char* CSG_UNDEFINED_     = "undefined" ; 


static const char* CSG_OBSOLETE_      = "obsolete" ; 
static const char* CSG_PARTLIST_      = "partlist" ; 
static const char* CSG_FLAGPARTLIST_  = "flagpartlist" ; 
static const char* CSG_FLAGNODETREE_  = "flagnodetree" ; 
static const char* CSG_FLAGINVISIBLE_ = "flaginvisible" ; 
static const char* CSG_PMT_           = "pmt" ; 
static const char* CSG_ZLENS_         = "zlens" ; 
static const char* CSG_PRISM_         = "prism" ; 
static const char* CSG_LAST_          = "last" ; 



struct CSG
{
    static OpticksCSG_t BooleanOperator(char op) 
    {   
        OpticksCSG_t typecode = CSG_ZERO ;   
        switch(op)
        {   
           case 'U':  typecode = CSG_UNION         ; break ; 
           case 'I':  typecode = CSG_INTERSECTION  ; break ; 
           case 'D':  typecode = CSG_DIFFERENCE    ; break ; 
        }   
        assert( typecode != CSG_ZERO );  
        return typecode ;   
    }   

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
        else if(strcmp(nodename, CSG_INFCYLINDER_) == 0)    tc = CSG_INFCYLINDER ;
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
        else if(strcmp(nodename, CSG_PHICUT_) == 0)         tc = CSG_PHICUT ;
        else if(strcmp(nodename, CSG_LPHICUT_) == 0)        tc = CSG_LPHICUT ;
        else if(strcmp(nodename, CSG_THETACUT_) == 0)       tc = CSG_THETACUT ;
        else if(strcmp(nodename, CSG_LTHETACUT_) == 0)      tc = CSG_LTHETACUT ;
        else if(strcmp(nodename, CSG_DISCONTIGUOUS_) == 0)   tc = CSG_DISCONTIGUOUS ;
        else if(strcmp(nodename, CSG_CONTIGUOUS_) == 0)      tc = CSG_CONTIGUOUS ;
        else if(strcmp(nodename, CSG_OVERLAP_) == 0)         tc = CSG_OVERLAP ;
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

    static const char* Name( unsigned type )
    {
        return Name((OpticksCSG_t)type); 
    }
    static const char* Name( OpticksCSG_t type )
    {
        const char* s = NULL ; 
        switch(type)
        {
            case CSG_ZERO:          s = CSG_ZERO_          ; break ; 

            case CSG_TREE:          s = CSG_TREE_          ; break ; 
            case CSG_UNION:         s = CSG_UNION_         ; break ; 
            case CSG_INTERSECTION:  s = CSG_INTERSECTION_  ; break ; 
            case CSG_DIFFERENCE:    s = CSG_DIFFERENCE_    ; break ; 

            case CSG_NODE:          s = CSG_NODE_          ; break ; 
            case CSG_CONTIGUOUS:    s = CSG_CONTIGUOUS_    ; break ; 
            case CSG_DISCONTIGUOUS: s = CSG_DISCONTIGUOUS_ ; break ; 
            case CSG_OVERLAP:       s = CSG_OVERLAP_       ; break ; 

            case CSG_LEAF:          s = CSG_LEAF_          ; break ; 
            case CSG_SPHERE:        s = CSG_SPHERE_        ; break ; 
            case CSG_BOX:           s = CSG_BOX_           ; break ; 
            case CSG_ZSPHERE:       s = CSG_ZSPHERE_       ; break ; 
            case CSG_TUBS:          s = CSG_TUBS_          ; break ; 
            case CSG_CYLINDER:      s = CSG_CYLINDER_      ; break ; 
            case CSG_SLAB:          s = CSG_SLAB_          ; break ; 
            case CSG_PLANE:         s = CSG_PLANE_         ; break ; 
            case CSG_CONE:          s = CSG_CONE_          ; break ; 
            case CSG_MULTICONE:     s = CSG_MULTICONE_     ; break ; 
            case CSG_BOX3:          s = CSG_BOX3_          ; break ; 
            case CSG_TRAPEZOID:     s = CSG_TRAPEZOID_     ; break ; 
            case CSG_CONVEXPOLYHEDRON: s = CSG_CONVEXPOLYHEDRON_ ; break ; 
            case CSG_DISC:          s = CSG_DISC_          ; break ; 
            case CSG_SEGMENT:       s = CSG_SEGMENT_       ; break ; 
            case CSG_ELLIPSOID:     s = CSG_ELLIPSOID_     ; break ; 
            case CSG_TORUS:         s = CSG_TORUS_         ; break ; 
            case CSG_HYPERBOLOID:   s = CSG_HYPERBOLOID_   ; break ; 
            case CSG_CUBIC:         s = CSG_CUBIC_         ; break ; 
            case CSG_INFCYLINDER:   s = CSG_INFCYLINDER_   ; break ; 
            case CSG_PHICUT:        s = CSG_PHICUT_        ; break ; 
            case CSG_LPHICUT:       s = CSG_LPHICUT_       ; break ; 
            case CSG_THETACUT:      s = CSG_THETACUT_      ; break ; 
            case CSG_LTHETACUT:     s = CSG_LTHETACUT_      ; break ; 
            case CSG_UNDEFINED:     s = CSG_UNDEFINED_     ; break ; 

            case CSG_OBSOLETE:      s = CSG_OBSOLETE_      ; break ; 
            case CSG_PARTLIST:      s = CSG_PARTLIST_      ; break ; 
            case CSG_FLAGPARTLIST:  s = CSG_FLAGPARTLIST_  ; break ; 
            case CSG_FLAGNODETREE:  s = CSG_FLAGNODETREE_  ; break ; 
            case CSG_FLAGINVISIBLE: s = CSG_FLAGINVISIBLE_ ; break ; 
            case CSG_PMT:           s = CSG_PMT_           ; break ; 
            case CSG_ZLENS:         s = CSG_ZLENS_         ; break ; 
            case CSG_PRISM:         s = CSG_PRISM_         ; break ; 
            case CSG_LAST:          s = CSG_LAST_          ; break ; 
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
        // hmm this is fine in context of tree : but are generalizing with lists as well as trees
        return !(type == CSG_INTERSECTION || type == CSG_UNION || type == CSG_DIFFERENCE) ; 
    }

    static bool IsTree(OpticksCSG_t type)
    {
        return (type == CSG_INTERSECTION || type == CSG_UNION || type == CSG_DIFFERENCE) ; 
    }
    // TODO: consolidate uses of these
    static bool IsOperator(OpticksCSG_t type)
    {
        return  (type == CSG_INTERSECTION || type == CSG_UNION || type == CSG_DIFFERENCE) ; 
    }


    static bool IsList(OpticksCSG_t type)
    {
        return  (type == CSG_CONTIGUOUS || type == CSG_DISCONTIGUOUS || type == CSG_OVERLAP ) ; 
    }

    static bool IsCompound(OpticksCSG_t type)
    { 
        return IsTree(type) || IsList(type) ; 
    }

    static bool IsLeaf(OpticksCSG_t type)
    {
        return !IsCompound(type) ;  
    }

    static bool IsUnbounded(OpticksCSG_t type)
    {
        return  type == CSG_PHICUT || type == CSG_THETACUT || type == CSG_INFCYLINDER  || type == CSG_PLANE || type == CSG_SLAB || type == CSG_LPHICUT || type == CSG_LTHETACUT ; 
    }

    static bool IsUnion(OpticksCSG_t type)
    {
        return  type == CSG_UNION  ; 
    }

    static bool IsIntersection(OpticksCSG_t type)
    {
        return  type == CSG_INTERSECTION  ; 
    }

    static bool IsDifference(OpticksCSG_t type)
    {
        return  type == CSG_DIFFERENCE  ; 
    }

    static bool IsZero(OpticksCSG_t type)
    {
        return  type == CSG_ZERO ; 
    }

    static unsigned UnionMask()
    {
        return 1 << CSG_UNION ; 
    }

    static unsigned IntersectionMask()
    {
        return 1 << CSG_INTERSECTION ; 
    }

    static unsigned DifferenceMask()
    {
        return 1 << CSG_DIFFERENCE ; 
    }

    static bool IsPositiveMask( unsigned mask )
    {
        return ( mask & DifferenceMask() ) == 0 ; 
    }

    static std::string TypeMask( unsigned mask )
    {
         std::stringstream ss ; 
         if((mask & UnionMask()) != 0)        ss << Tag(CSG_UNION) << " " ; 
         if((mask & IntersectionMask()) != 0) ss << Tag(CSG_INTERSECTION) << " " ; 
         if((mask & DifferenceMask()) != 0)   ss << Tag(CSG_DIFFERENCE) << " " ; 
         std::string s = ss.str(); 
         return s ; 
    }

    static bool HasPlanes(OpticksCSG_t type)
    {
        return (type == CSG_TRAPEZOID || type == CSG_CONVEXPOLYHEDRON || type == CSG_SEGMENT ) ; 
    }

    static bool HasPlanes(unsigned type)
    {
         return HasPlanes((OpticksCSG_t)type); 
    }

};


#endif

