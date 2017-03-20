#pragma once

#include "NNode.hpp"
#include "NPart.hpp"

#include "NPY_API_EXPORT.hh"

struct NPY_API nbox : nnode {

    // NO CTOR
    double operator()(double px, double py, double pz) ;
    nbbox bbox();

};

// only methods that are specific to boxes 
// and need to override the nnode need to be here 


inline NPY_API nbox make_nbox(float x, float y, float z, float w)
{
    nbox n ; nnode::Init(n, CSG_BOX) ; n.param.x = x ; n.param.y = y ; n.param.z = z ; n.param.w = w ; return n ;
}

inline NPY_API nbox make_nbox(const nvec4& p)
{
    nbox n ; nnode::Init(n,CSG_BOX) ; n.param = p ; return n ;
}
inline NPY_API nbox* make_nbox_ptr(const nvec4& p)
{
    nbox* n = new nbox ; nnode::Init(*n,CSG_BOX) ; n->param = p ; return n ; 
}
inline NPY_API nbox* make_nbox_ptr(float x, float y, float z, float w)
{
    nbox* n = new nbox ; nnode::Init(*n,CSG_BOX) ; n->param.x = x ; n->param.y = y ; n->param.z = z ; n->param.w = w ; return n ; 
}



