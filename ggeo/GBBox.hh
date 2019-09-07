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


// TODO: get rid of this, move to nbbox 
// not so easy to jump to nbbox or adopt glm::vec3 here 
// due to stuffing into buffers requirements of GMesh 

#include "GVector.hh"
#include "GGEO_API_EXPORT.hh"

struct nbbox ; 

struct GGEO_API gbbox 
{
   gbbox() : min(gfloat3(0.f)), max(gfloat3(0.f)) {} ;
   gbbox(float s) :  min(gfloat3(-s)), max(gfloat3(s)) {} ; 
   gbbox(const gfloat3& _min, const gfloat3& _max) :  min(_min), max(_max) {} ; 

   gbbox(const gbbox& other ) : min(other.min), max(other.max) {} ;
   gbbox(const nbbox& other );

   gfloat3 dimensions();
   gfloat3 center();
   void enlarge(float factor);  //  multiple of extent
   void include(const gbbox& other);
   gbbox& operator *= (const GMatrixF& m) ;
   float extent(const gfloat3& dim);
   gfloat4 center_extent();

   void Summary(const char* msg) const ;
   std::string description() const ;
   std::string desc() const ;

   // stuffing gbbox into GBuffer makes it not so straightforward to move to glm::vec3 
   gfloat3 min  ; 
   gfloat3 max  ; 


};





