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
#include "NGLM.hpp"

struct npart ;
struct nbbox ; 
struct nuv ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API ncone : nnode 
{
    float operator()(float x, float y, float z) const ;
    nbbox bbox() const;
    //npart part() const ;

    glm::vec3 gseedcenter() const  ;
    glm::vec3 gseeddir() const ;
    void pdump(const char* msg="ncone::pdump") const ;


    int       par_euler() const ; 
    unsigned  par_nsurf() const ; 
    glm::vec3 par_pos_model(const nuv& uv) const  ;
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ; 
    static void _par_pos_body(glm::vec3& pos,  const nuv& uv, const float r1_, const float z1_,  const float r2_, const float z2_);  

 
    glm::vec3 center() const  ; 
    glm::vec2 cnormal() const  ; 
    glm::vec2 csurface() const  ; 

    void increase_z2(float dz);
    void decrease_z1(float dz);

    float x() const ; 
    float y() const ; 

    float r1() const ; 
    float z1() const ; 
    float r2() const  ; 
    float z2() const  ; 

    float rmax() const  ; 
    float zc() const  ; 
    float z0() const  ; // apex
    float tantheta() const  ; 
};



inline NPY_API float ncone::r1() const { return param.f.x ; }
inline NPY_API float ncone::z1() const { return param.f.y ; }
inline NPY_API float ncone::r2() const { return param.f.z ; }
inline NPY_API float ncone::z2() const { return param.f.w ; }  // z2 > z1

// grow the cone on upwards on upper side (z2) or downwards on down side (z1)
inline NPY_API void  ncone::increase_z2(float dz){ assert( dz >= 0.f) ; param.f.w += dz ; } // z2 > z1
inline NPY_API void  ncone::decrease_z1(float dz){ assert( dz >= 0.f) ; param.f.y -= dz ; }

inline NPY_API float ncone::zc() const { return (z1() + z2())/2.f ; }
inline NPY_API float ncone::rmax() const { return fmaxf( r1(), r2())  ; }
inline NPY_API float ncone::z0() const {  return (z2()*r1()-z1()*r2())/(r1()-r2()) ; }
inline NPY_API float ncone::tantheta() const { return (r2()-r1())/(z2()-z1()) ; }
inline NPY_API float ncone::x() const { return 0.f ; }
inline NPY_API float ncone::y() const { return 0.f ; }
inline NPY_API glm::vec3 ncone::center() const { return glm::vec3(x(),y(),zc()) ; } 
inline NPY_API glm::vec2 ncone::cnormal() const { return glm::normalize( glm::vec2(z2()-z1(),r1()-r2()) ) ; }
inline NPY_API glm::vec2 ncone::csurface() const { glm::vec2 cn = cnormal() ; return glm::vec2( cn.y, -cn.x ) ; }      


inline NPY_API void init_cone(ncone* n, const nquad& param)
{
    n->param = param ;
    assert( n->z2() > n->z1() );
}

inline NPY_API ncone* make_cone(const nquad& param)
{
    ncone* n = new ncone ; 
    nnode::Init(n,CSG_CONE) ; 
    init_cone(n, param);
    return n ; 
}

inline NPY_API ncone* make_cone(float r1, float z1, float r2, float z2)
{
    nquad param ;
    param.f = {r1,z1,r2,z2} ;
    return make_cone(param);
}


