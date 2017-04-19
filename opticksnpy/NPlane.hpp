#pragma once

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
    float operator()(float x, float y, float z) ;

    bool intersect( const float tmin, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect );

    glm::vec3 gseedcenter();
    glm::vec3 gseeddir();
    void pdump(const char* msg="nplane::dump", int verbosity=1);

    glm::vec3 n ;  // normal
    float     d ;  // signed distance to origin
};


inline NPY_API void init_plane(nplane& plane, const nquad& param )
{
    glm::vec3 n = glm::normalize(glm::vec3(param.f.x, param.f.y, param.f.z));

    plane.n = n ; 
    plane.d = param.f.w ; 

    plane.param.f.x = n.x ;
    plane.param.f.y = n.y ;
    plane.param.f.z = n.z ;
    plane.param.f.w = plane.d  ;

}
inline NPY_API nplane make_plane(const nquad& param)
{  
    nplane plane ; 
    nnode::Init(plane,CSG_PLANE) ; 
    init_plane(plane, param );
    return plane ;
}
inline NPY_API nplane make_plane(float x, float y, float z, float w)
{
    nquad param ;  
    param.f = {x,y,z,w} ;
    return make_plane( param ); 
}







struct NPY_API ndisc {
    float z() const;
    nplane plane ;
    float radius ;  

    void dump(const char* msg);
};


inline NPY_API ndisc make_disc(const nplane& plane_, float radius_) 
{
   ndisc d ; d.plane = plane_ ; d.radius = radius_ ; return d ; 
}




