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

#include "NGLM.hpp"
#include "NNode.hpp"

#include "NPY_API_EXPORT.hh"

struct NPY_API nconvexpolyhedron : nnode 
{
    float operator()(float x, float y, float z) const ;

    bool intersect( const float tmin, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect ) const ;
    void check_planes() const ;   

    nbbox bbox() const ;
    nbbox bbox_model() const ;
    glm::vec3 gseedcenter();
    glm::vec3 gseeddir();

    glm::vec3 par_pos_model(const nuv& uv) const  ;

    // TODO: produce some more par points by using similar 
    //       basis finding approach of nslab to populate a 
    //       plane : and select points by SDF


    unsigned  par_nsurf() const ; 
    int       par_euler() const ; 
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ; 

    void set_planes(const std::vector<glm::vec4>& planes_) ;
    void set_srcvertsfaces( const std::vector<glm::vec3>& srcverts_ , const std::vector<glm::ivec4>& srcfaces_ ) ;
    void dump_srcvertsfaces() const ; 


    void set_bbox(const nbbox& bb) ;

    static nconvexpolyhedron* make_trapezoid_cube();
    static nconvexpolyhedron* make_trapezoid(float z, float x1, float y1, float x2, float y2 );   
    static nconvexpolyhedron* make_segment(float phi0, float phi1, float sz, float sr ) ;


    nconvexpolyhedron* make_transformed( const glm::mat4& t ) const ;

    void pdump(const char* msg="nconvexpolyhedron::pdump") const ;


    void define_uv_basis();
    void dump_uv_basis(const char* msg="nconvexpolyhedron::dump_uv_basis") const; 

    std::vector<glm::vec3> udirs ; 
    std::vector<glm::vec3> vdirs ; 

    std::vector<glm::vec3>  srcverts ; 
    std::vector<glm::ivec4> srcfaces ; 


};

inline NPY_API void init_convexpolyhedron(nconvexpolyhedron* n, const nquad& param, const nquad& param1, const nquad& param2, const nquad& param3 )
{
    n->param = param ; 
    n->param1 = param1 ; 
    n->param2 = param2 ; 
    n->param3 = param3 ; 
}

inline NPY_API nconvexpolyhedron* make_convexpolyhedron(const nquad& param, const nquad& param1, const nquad& param2, const nquad& param3)
{
    nconvexpolyhedron* n = new nconvexpolyhedron ; 
    nnode::Init(n,CSG_CONVEXPOLYHEDRON) ; 
    init_convexpolyhedron(n, param, param1, param2, param3 );
    return n ;
}

/*
inline NPY_API nconvexpolyhedron* make_convexpolyhedron_ptr(const nquad& param, const nquad& param1, const nquad& param2, const nquad& param3)
{
    nconvexpolyhedron* cpol = new nconvexpolyhedron ; 
    nnode::Init(*cpol,CSG_CONVEXPOLYHEDRON) ; 
    init_convexpolyhedron(*cpol, param, param1, param2, param3 );
    return cpol ;
}
*/


inline NPY_API nconvexpolyhedron* make_convexpolyhedron()
{
    nquad param, param1, param2, param3 ; 
    param.u = {0,0,0,0} ;
    param1.u = {0,0,0,0} ;
    param2.u = {0,0,0,0} ;
    param3.u = {0,0,0,0} ;
    return make_convexpolyhedron(param, param1, param2, param3 );
}

/*
inline NPY_API nconvexpolyhedron* make_convexpolyhedron_ptr()
{
    nquad param, param1, param2, param3 ; 
    param.u = {0,0,0,0} ;
    param1.u = {0,0,0,0} ;
    param2.u = {0,0,0,0} ;
    param3.u = {0,0,0,0} ;
    return make_convexpolyhedron_ptr(param, param1, param2, param3 );
}
*/

