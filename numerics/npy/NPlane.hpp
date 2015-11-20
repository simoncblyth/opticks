#pragma once

#include "NQuad.hpp"

struct nbbox {
    // ctor assuming rotational symmetry around z axis
    nbbox(float zmin, float zmax, float ymin, float ymax); 
    void dump(const char* msg);

    nvec4 min ; 
    nvec4 max ; 
};


inline nbbox::nbbox(float zmin, float zmax, float ymin, float ymax)
{
    min.x = ymin ; 
    min.y = ymin ; 
    min.z = zmin ;
    min.w = 0 ;

    max.x = ymax ; 
    max.y = ymax ; 
    max.z = zmax ;
    max.w = 0 ;
}


struct nplane {
    // http://mathworld.wolfram.com/Plane.html
    // xyz: normalized normal vector, w:distance from origin

    nplane(float x, float y, float z, float w);
    nplane(const nvec4& param_);
    void dump(const char* msg);

    nvec4 param ; 
};

inline nplane::nplane(float x, float y, float z, float w)
{
    param.x = x  ;
    param.y = y  ;
    param.z = z  ;
    param.w = w  ;
}

inline nplane::nplane(const nvec4& param_)
{
    param = param_ ;
}


struct ndisc {
    ndisc( const nplane& plane_, float radius_ );
    float z() const;
    nplane plane ;
    float radius ;  

    void dump(const char* msg);
};

inline ndisc::ndisc(const nplane& plane_, float radius_) 
    :
    plane(plane_),
    radius(radius_)
{
}


inline float ndisc::z() const 
{
   return plane.param.w ;  
}




