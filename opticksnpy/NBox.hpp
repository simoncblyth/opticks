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
    nbox n ; nnode::Init(n, CSG_BOX) ; n.param.x = x ; n.param.y = y ; n.param.z = z ; n.param.w = w ; return n ;
}

inline NPY_API nbox make_nbox(const nvec4& p)
{
    nbox n ; nnode::Init(n,CSG_BOX) ; n.param.x = p.x ; n.param.y = p.y ; n.param.z = p.z ; n.param.w = p.w ; return n ;
}


