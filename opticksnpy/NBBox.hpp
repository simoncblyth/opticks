#pragma once

#include "NQuad.hpp"
#include "NPY_API_EXPORT.hh"


struct NPY_API nbbox {

    // NO CTOR

    void dump(const char* msg);
    void include(const nbbox& other );

    nvec3 min ; 
    nvec3 max ; 
};


// "ctor" assuming rotational symmetry around z axis
inline NPY_API nbbox make_nbbox(float zmin, float zmax, float ymin, float ymax)
{
    nbbox bb ; 
    bb.min = make_nvec3( ymin, ymin, zmin ) ;
    bb.max = make_nvec3( ymax, ymax, zmax ) ;
    return bb ;
}



inline NPY_API nbbox make_nbbox()
{
    return make_nbbox(0,0,0,0) ;
}


