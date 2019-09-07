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

#include "NNode.hpp"
#include "NQuad.hpp"
#include "NGLM.hpp"

struct npart ;
struct nbbox ; 
struct nuv ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API ntorus : nnode 
{
    float operator()(float x, float y, float z) const ;
    nbbox bbox() const ;

    glm::vec3 gseeddir();
    glm::vec3 gseedcenter() const ;

    unsigned  par_nsurf() const ; 
    glm::vec3 par_pos_model(const nuv& uv) const  ;
    int       par_euler() const ; 
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ; 

    static void _par_pos_body(glm::vec3& pos,  const nuv& uv, const float R_, const float r_ ) ;

    void pdump(const char* msg="ntorus::pdump") const ;

    float rmajor() const ; 
    float rminor() const ; 
 
    glm::vec3 center() const  ; 

};


inline NPY_API glm::vec3 ntorus::center() const { return glm::vec3(0,0,0)  ; }

inline NPY_API float ntorus::rminor() const { return param.f.z  ; }
inline NPY_API float ntorus::rmajor() const { return param.f.w  ; }




inline NPY_API void init_torus(ntorus* n, const nquad& param)
{
    n->param = param ; 
}
inline NPY_API ntorus* make_torus(const nquad& param)
{
    ntorus* n = new ntorus ; 
    nnode::Init(n,CSG_TORUS) ; 
    init_torus(n, param);
    return n ; 
}
inline NPY_API ntorus* make_torus(float x, float y, float z, float w)
{
    nquad param ; 
    param.f = {x,y,z,w} ;
    return make_torus(param);
}
inline NPY_API ntorus* make_torus(float R=100.f, float r=10.f)
{
    return make_torus(0.f,0.f,r,R);
}



