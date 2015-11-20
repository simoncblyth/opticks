#pragma once

#include <glm/glm.hpp>


struct nbbox {
    // ctor assuming rotational symmetry around z axis
    nbbox(float zmin, float zmax, float ymin, float ymax); 
    void dump(const char* msg);

    glm::vec4 min ; 
    glm::vec4 max ; 
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
    nplane(const glm::vec4& param_);
    void dump(const char* msg);

    glm::vec4 param ; 
};

inline nplane::nplane(float x, float y, float z, float w)
{
    param.x = x  ;
    param.y = y  ;
    param.z = z  ;
    param.w = w  ;
}

inline nplane::nplane(const glm::vec4& param_)
{
    param = param_ ;
}




struct ndisc {
    ndisc( const nplane& plane_, float radius_ );
    void dump(const char* msg);
    float z();

    nplane plane ;
    float radius ;  
};

inline ndisc::ndisc(const nplane& plane_, float radius_) 
    :
    plane(plane_),
    radius(radius_)
{
}


inline float ndisc::z()
{
   return plane.param.w ;  
}



