#pragma once

#include "NGLM.hpp"
#include "NNode.hpp"
#include "NPart.hpp"

#include "NPY_API_EXPORT.hh"

struct NPY_API nbox : nnode {

    // NO CTOR
    double operator()(double px, double py, double pz) ;
    nbbox bbox();

    glm::vec3 center ; 

};

// only methods that are specific to boxes 
// and need to override the nnode need to be here 



inline NPY_API void init_nbox(nbox& b, const nvec4& p )
{
    b.param = p ; 
    b.center.x = p.x ; 
    b.center.y = p.y ; 
    b.center.z = p.z ; 
}

inline NPY_API nbox make_nbox(const nvec4& p)
{
    nbox n ; 
    nnode::Init(n,CSG_BOX) ; 
    init_nbox(n, p );
    return n ;
}
inline NPY_API nbox* make_nbox_ptr(const nvec4& p)
{
    nbox* n = new nbox ; 
    nnode::Init(*n,CSG_BOX) ; 
    init_nbox(*n, p );
    return n ; 
}

inline NPY_API nbox make_nbox(float x, float y, float z, float w)
{
    nvec4 param = {x,y,z,w} ;
    return make_nbox( param ); 
}
inline NPY_API nbox* make_nbox_ptr(float x, float y, float z, float w)
{
    nvec4 param = {x,y,z,w} ;
    return make_nbox_ptr( param ); 
}



