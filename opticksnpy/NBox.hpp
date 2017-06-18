#pragma once

#include "NGLM.hpp"
#include "NNode.hpp"
#include "NPart.hpp"

#include "NPY_API_EXPORT.hh"

/*

nbox 
======

Currently two flavors of box use nbox class

#. CSG_BOX (positioned box with single dimension control) 
#. CSG_BOX3 (unplaced box with 3 dimension control) 

Perhaps CSG_BOX3 will usurp CSG_BOX at which point can 
change the enum.

*/

struct nmat4triple ; 

struct NPY_API nbox : nnode 
{
    //  geometry modifiers

    void adjustToFit(const nbbox& container_bb, float scale);
    void nudge(unsigned s, float delta);

    // signed distance functions

    float operator()(float x, float y, float z) const ;

    float sdf_(const glm::vec3& pos, NNodeFrameType fr) const ;
    float sdf_(const glm::vec3& pos, const nmat4triple* triple) const ; 
    float sdf_model(const glm::vec3& pos) const ; 
    float sdf_local(const glm::vec3& pos) const ; 
    float sdf_global(const glm::vec3& pos) const ; 

    // testing sdf imps

    float sdf1(float x, float y, float z) const ;
    float sdf2(float x, float y, float z) const ;

    // bounding box

    nbbox bbox_(NNodeFrameType fty) const ;
    nbbox bbox_(const nmat4triple* triple) const ;
    nbbox bbox_model() const ;
    nbbox bbox_local() const ;
    nbbox bbox_global() const ;
    nbbox bbox() const ;

    // parametric surface positions 

    glm::vec3 par_pos_(const nuv& uv, NNodeFrameType fr) const ;   
    glm::vec3 par_pos_(const nuv& uv, const nmat4triple* triple) const ;   
    glm::vec3 par_pos_model(const nuv& uv) const ;  // no transforms, bare model
    glm::vec3 par_pos_local(const nuv& uv) const ;  // "transform"  local node frame
    glm::vec3 par_pos_global(const nuv& uv) const ; // "gtransform" CSG tree root node frame 
    glm::vec3 par_pos(const nuv& uv) const ;        // "gtransform" CSG tree root node frame 

    // parametric metadata

    unsigned  par_nsurf() const ;
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ;
    int       par_euler() const ; 
 
    // seedcenter needed for ImplicitMesher

    glm::vec3 gseedcenter() const ;

    void pdump(const char* msg="nbox::pdump") const ;


    glm::vec3 center ; 
    bool is_box3 ; 

};

// only methods that are specific to boxes 
// and need to override the nnode need to be here 




inline NPY_API void init_box(nbox& b, const nquad& p )
{
    b.param = p ; 
    b.center.x = p.f.x ; 
    b.center.y = p.f.y ; 
    b.center.z = p.f.z ; 
    b.is_box3 = false ; 
}
inline NPY_API void init_box3(nbox& b, const nquad& p )
{
    b.param = p ; 
    b.center.x = 0 ; 
    b.center.y = 0 ; 
    b.center.z = 0 ; 
    b.is_box3 = true ; 
}


inline NPY_API nbox make_box(const nquad& p)
{
    nbox n ; 
    nnode::Init(n,CSG_BOX) ; 
    init_box(n, p );
    return n ;
}
inline NPY_API nbox make_box3(const nquad& p)
{
    nbox n ; 
    nnode::Init(n,CSG_BOX3) ; 
    init_box3(n, p );
    return n ;
}


inline NPY_API nbox make_box(float x, float y, float z, float w)  // center and halfside
{
    nquad param ;
    param.f =  {x,y,z,w} ;
    return make_box( param ); 
}
inline NPY_API nbox make_box3(float x, float y, float z) // three 
{
    nquad param ;
    param.f =  {x,y,z,0} ;
    return make_box3( param ); 
}





