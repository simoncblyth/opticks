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
#include <string>
#include "NPY_API_EXPORT.hh"

struct NPY_API nuv 
{
    unsigned s() const {   return s_ & 0xffff ; } 
    unsigned u() const {   return u_ & 0xffff ; } 
    unsigned v() const {   return v_ & 0xffff ; } 
    unsigned p() const {   return p_ ; }   // primitive index

    unsigned ps() const {  return p()*100 + s() ; } 

    static unsigned ps_to_prim( unsigned ps_){ return ps_/100   ; }
    static unsigned ps_to_sheet(unsigned ps_){ return ps_ % 100 ; }

    bool matches(unsigned prim, unsigned sheet) const { return prim == p() && sheet == s() ; }


    unsigned nu() const {  return u_ >> 16 ; } 
    unsigned nv() const {  return v_ >> 16 ; } ;

    float   fu() const  { return float(u())/float(nu()) ; } 
    float   fv() const  { return float(v())/float(nv()) ; } 

    float  fu2pi(bool seamed=true) const ; 
    float  fv2pi(bool seamed=true) const ; 
    float  fvpi() const ; 

    bool is_interior(unsigned margin=0) const ;


    std::string desc() const ;
    std::string detail() const ;

    unsigned s_ ; 
    unsigned u_ ; 
    unsigned v_ ;
    unsigned p_ ;
};

/**
make_uv
=========

s
   surface index, for example a cube has 6 surface "sheets" with s values 0,1,2,3,4,5
u,v
   2D parametric surface indices indicating a point on the surface 
nu,nv
   number of 2D surface indices 
p
   prim index (?)


**/

inline nuv make_uv(unsigned s, unsigned u, unsigned v, unsigned nu, unsigned nv, unsigned p)
{
   nuv uv ; 

   uv.s_ = s ; 
   uv.u_ = (nu << 16) | u  ;
   uv.v_ = (nv << 16) | v  ;
   uv.p_ = p ; 


   return uv ; 
}

