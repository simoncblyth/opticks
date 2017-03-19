#pragma once

#include "NQuad.hpp"
#include "NNode.hpp"
#include "NPart.hpp"

#include "NPY_API_EXPORT.hh"

struct NPY_API nbox : nnode {

    // NO CTOR

    double operator()(double px, double py, double pz) ;
    nbbox bbox();
 
    npart part();
    void dump(const char* msg="nbox::dump");

    nvec4 param ; 
};


inline NPY_API nbox make_nbox(float x, float y, float z, float w)
{
    nbox b ; b.type = CSG_BOX ; b.param.x = x ; b.param.y = y ; b.param.z = z ; b.param.w = w ; return b ;
}

inline NPY_API nbox make_nbox(const nvec4& p)
{
    nbox b ; b.type = CSG_BOX ; b.param.x = p.x ; b.param.y = p.y ; b.param.z = p.z ; b.param.w = p.w ; return b ;
}


