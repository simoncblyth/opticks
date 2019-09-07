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

/*
https://www.math.hmc.edu/~gu/curves_and_surfaces/surfaces/hyperboloid.html
*/

#include "NNode.hpp"
#include "NQuad.hpp"
#include "NGLM.hpp"

struct npart ;
struct nbbox ; 
struct nuv ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API nhyperboloid : nnode 
{
    float operator()(float x, float y, float z) const ;


    float r0() const ;
    float zf() const ;
    float z1() const ;
    float z2() const ;

    float rrz(float z) const ;
    float  rz(float z) const ;

    nbbox bbox() const ;
    glm::vec3 gseedcenter() const ;

    unsigned  par_nsurf() const ; 
    glm::vec3 par_pos_model(const nuv& uv) const  ;
    int       par_euler() const ; 
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ; 

    static void _par_pos_body(glm::vec3& pos,  const nuv& uv, const float r_ ) ;

    void pdump(const char* msg="nhyperboloid::pdump") const ;
 
    glm::vec3 center() const  ; 

};

inline NPY_API float nhyperboloid::r0() const { return param.f.x ; }
inline NPY_API float nhyperboloid::zf() const { return param.f.y ; }
inline NPY_API float nhyperboloid::z1() const { return param.f.z ; }
inline NPY_API float nhyperboloid::z2() const { return param.f.w ; }

inline NPY_API glm::vec3 nhyperboloid::center() const { return glm::vec3(0,0,0)  ; }


inline NPY_API void init_hyperboloid(nhyperboloid* n, const nquad& param)
{
    n->param = param ; 
}
inline NPY_API nhyperboloid* make_hyperboloid(const nquad& param)
{
    nhyperboloid* n = new nhyperboloid ; 
    nnode::Init(n,CSG_HYPERBOLOID) ; 
    init_hyperboloid(n, param);
    return n ; 
}

inline NPY_API nhyperboloid* make_hyperboloid(float r0=100.f, float zf=100.f, float z1=-100.f, float z2=100.f)
{
    nquad param ; 
    param.f = {r0,zf,z1,z2} ;
    return make_hyperboloid(param);
}


