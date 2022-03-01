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

#include <cassert>

#include "NNode.hpp"
#include "NQuad.hpp"
#include "NGLM.hpp"

//#include "NZSphere.h"

struct nbbox ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API nzsphere : nnode 
{
    static nzsphere* Create(float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1 ); 
    static nzsphere* Create(); 
    static nzsphere* Create(float x, float y, float z, float radius, float z1, float z2 ); 
    static nzsphere* Create(const nquad& param, const nquad& param1 ); 

    float operator()(float x, float y, float z) const ;

    nbbox bbox() const ;

    glm::vec3 gseedcenter() const  ;
    glm::vec3 gseeddir() const ;

    void pdump(const char* msg="nzsphere::pdump") const ;

    unsigned  par_nsurf() const ; 
    glm::vec3 par_pos_model(const nuv& uv) const  ;
    int       par_euler() const ; 
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ; 

    static void _par_pos_body(glm::vec3& pos,  const nuv& uv, const float r_, const float z1_, const float z2_, const bool has_z1cap_ , const bool has_z2cap_  ) ;

    bool      has_z1_endcap() const ;
    bool      has_z2_endcap() const ;



    unsigned flags() const ;

    float     x() const ; 
    float     y() const ; 
    float     z() const ; 
    float     radius() const  ; 
    float     r1() const  ;    // of the endcaps at z1,z2  
    float     r2() const  ; 
    float     rz(float z) const ; // radius of z-slice endcap

    float  startTheta() const ; 
    float  endTheta() const ; 
    float  deltaTheta() const ; 

    glm::vec3 center() const  ; 

    float     z2() const ; 
    float     z1() const ; 

    float zmax() const ;
    float zmin() const ;
    float zc() const ;


    void   check() const ; 

    void increase_z2(float dz);
    void decrease_z1(float dz);
    void set_zcut(float z1, float z2);


};


