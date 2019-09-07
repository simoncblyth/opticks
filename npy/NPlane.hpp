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

#include "NGLM.hpp"
#include "NQuad.hpp"
#include "NNode.hpp"

#include "NPY_API_EXPORT.hh"

/*

http://mathworld.wolfram.com/Plane.html
xyz: normalized normal vector, w:distance from origin

RTCD 

p116: Kay-Kajiya slab based volumes
p126: Closest point on plane to a point in space



                          +ve 
              n ^
                |
      ----------+--------- 0 -
        |
        d                 -ve      
        |
      ----------O-------------

*/



struct NPY_API nplane : nnode 
{
    float operator()(float x, float y, float z) const ;

    bool intersect( const float tmin, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect );
    std::string desc() const  ; 

    glm::vec3 gseedcenter();
    glm::vec3 gseeddir();
    void pdump(const char* msg="nplane::dump") const ;

    glm::vec4 make_transformed(const glm::mat4& t) const ;

    glm::vec3 point_in_plane() const ;
    glm::vec4 get() const ;  
    glm::vec3 normal() const ;  
    float     distance_to_origin() const ; // signed 
};


inline NPY_API float     nplane::distance_to_origin() const { return param.f.w ; }
inline NPY_API glm::vec3 nplane::normal() const { return glm::vec3(param.f.x,param.f.y,param.f.z); }
inline NPY_API glm::vec4 nplane::get() const {    return glm::vec4(normal(),distance_to_origin()) ; }


inline NPY_API void init_plane(nplane* n, const nquad& param )
{
    glm::vec3 nrm = glm::normalize(glm::vec3(param.f.x, param.f.y, param.f.z));

    n->param.f.x = nrm.x ;
    n->param.f.y = nrm.y ;
    n->param.f.z = nrm.z ;
    n->param.f.w = param.f.w  ;

}
inline NPY_API nplane* make_plane(const nquad& param)
{  
    nplane* n = new nplane ; 
    nnode::Init(n,CSG_PLANE) ; 
    init_plane(n, param );
    return n ;
}
inline NPY_API nplane* make_plane(float x, float y, float z, float w)
{
    nquad param ;  
    param.f = {x,y,z,w} ;
    return make_plane( param ); 
}

inline NPY_API nplane* make_plane(const glm::vec4& par)
{
    nquad param ;  
    param.f = {par.x,par.y,par.z,par.w} ;
    return make_plane( param ); 
}



inline NPY_API glm::vec3 make_normal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{   
    glm::vec3 ba = b - a ; 
    glm::vec3 ca = c - a ; 
    glm::vec3 xx = glm::cross(ba, ca );
    return glm::normalize(xx) ; 
} 

inline NPY_API glm::vec4 make_plane(const glm::vec3& normal, const glm::vec3 point)
{
    glm::vec3 n = glm::normalize(normal);
    float     d = glm::dot(n, point);
    return glm::vec4(n, d);
}

inline NPY_API glm::vec4 make_plane(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{
    glm::vec3 n = make_normal( a, b, c );
    return make_plane(n, a);
}





struct NPY_API ndisk { // NB *ndisk* is not the same as *ndisc* (degenerated ncylinder)
    float z() const;
    nplane plane ;
    float radius ;  

    void dump(const char* msg);
};


inline NPY_API ndisk* make_disk(const nplane* plane_, float radius_) 
{
   assert( plane_ ); 
   ndisk* n = new ndisk ; n->plane = *plane_ ; n->radius = radius_ ; return n ; 
}




