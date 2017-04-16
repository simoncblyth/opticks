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

    glm::vec3 gcenter();
    void pdump(const char* msg="nplane::dump", int verbosity=1);

    glm::vec3 n ;  // normal
    float     d ;  // signed distance to origin
};


inline NPY_API void init_nplane(nplane& plane, const nvec4& param )
{
    glm::vec3 n = glm::normalize(glm::vec3(param.x, param.y, param.z));

    plane.n = n ; 
    plane.d = param.w ; 

    plane.param.x = n.x ;
    plane.param.y = n.y ;
    plane.param.z = n.z ;
    plane.param.w = plane.d  ;

}
inline NPY_API nplane make_nplane(const nvec4& param)
{  
    nplane plane ; 
    nnode::Init(plane,CSG_PLANE) ; 
    init_nplane(plane, param );
    return plane ;
}
inline NPY_API nplane make_nplane(float x, float y, float z, float w)
{
    nvec4 param = {x,y,z,w} ;
    return make_nplane( param ); 
}







struct NPY_API ndisc {
    float z() const;
    nplane plane ;
    float radius ;  

    void dump(const char* msg);
};


inline NPY_API ndisc make_ndisc(const nplane& plane_, float radius_) 
{
   ndisc d ; d.plane = plane_ ; d.radius = radius_ ; return d ; 
}




