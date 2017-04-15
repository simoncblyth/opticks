#pragma once

#include "NGLM.hpp"
#include "NNode.hpp"

#include "NPY_API_EXPORT.hh"

struct NPY_API nslab : nnode 
{
    float operator()(float x, float y, float z) ;

    glm::vec3 gcenter();
    void pdump(const char* msg="nslab::pdump", int verbosity=1);


    glm::vec3 normal ; 
    float offset ; 

};


// only methods that are specific to slabs
// and need to override the nnode need to be here 

inline NPY_API void init_nslab(nslab& n, const nvec4& p )
{
    n.normal.x = p.x ; 
    n.normal.y = p.y ; 
    n.normal.z = p.z ; 
    n.offset   = p.w ; 
}

inline NPY_API nslab make_nslab(const nvec4& p)
{
    nslab n ; 
    nnode::Init(n,CSG_SLAB) ; 
    init_nslab(n, p );
    return n ;
}
inline NPY_API nslab* make_nslab_ptr(const nvec4& p)
{
    nslab* n = new nslab ; 
    nnode::Init(*n,CSG_SLAB) ; 
    init_nslab(*n, p );
    return n ; 
}

inline NPY_API nslab make_nslab(float x, float y, float z, float w)
{
    nvec4 param = {x,y,z,w} ;
    return make_nslab( param ); 
}
inline NPY_API nslab* make_nslab_ptr(float x, float y, float z, float w)
{
    nvec4 param = {x,y,z,w} ;
    return make_nslab_ptr( param ); 
}


