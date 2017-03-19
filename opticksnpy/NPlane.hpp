#pragma once

#include "NQuad.hpp"
#include "NPY_API_EXPORT.hh"



struct NPY_API nplane {
    // http://mathworld.wolfram.com/Plane.html
    // xyz: normalized normal vector, w:distance from origin

    void dump(const char* msg);

    nvec4 param ; 
};


inline NPY_API nplane make_nplane(float x, float y, float z, float w)
{  
   nplane pl ; pl.param.x = x ; pl.param.y = y ; pl.param.z = z ; pl.param.w = w ; return pl ; 
}

inline NPY_API nplane make_nplane(const nvec4& p)
{  
   nplane pl ; pl.param.x = p.x ; pl.param.y = p.y ; pl.param.z = p.z ; pl.param.w = p.w ; return pl ; 
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




