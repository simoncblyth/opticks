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


*/


struct NPY_API nplane : nnode 
{
    float operator()(float x, float y, float z) ;

    void dump(const char* msg);

    nvec4 param ; 
    glm::vec3 n ;  // normal
    float     d ;  // signed distance to origin
};


inline NPY_API void init_nplane(nplane& plane, const nvec4& param )
{
    plane.param = param ;
    plane.n.x = param.x ; 
    plane.n.y = param.y ; 
    plane.n.z = param.z ; 
    plane.d   = param.w ; 
}
inline NPY_API nplane make_nplane(const nvec4& param)
{  
   nplane plane ; 
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




