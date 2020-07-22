/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include "NGLM.hpp"
#include "NNode.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

/*

nbox 
======

Currently two flavors of box use nbox class

#. CSG_BOX (positioned box with single dimension control) 
#. CSG_BOX3 (unplaced box with 3 dimension control) 
#. TODO: CSG_BOX4 (box with z1/z2 controls)

*/

struct nmat4triple ; 



struct NPY_API nbox : nnode 
{
    //  geometry modifiers

    void resizeToFit(const nbbox& container_bb, float scale, float delta );
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

    // parametric surface positions and outward normals

    glm::vec3 par_pos_model(const nuv& uv) const ;  // no transforms, bare model
    
    //void par_posnrm_model(glm::vec3& pos, glm::vec3& nrm, const nuv& uv) const ;  
    void par_posnrm_model(glm::vec3& pos, glm::vec3& nrm, unsigned s, float fu, float fv) const ;  

    unsigned  par_nsurf() const ;
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ;
    int       par_euler() const ; 
 
    // seedcenter needed for ImplicitMesher
    glm::vec3 gseedcenter() const ;
    void pdump(const char* msg="nbox::pdump") const ;

    float x() const ; 
    float y() const ; 
    float z() const ; 
    glm::vec3 center() const  ; 
    bool is_centered() const  ;   // always true for BOX3, sometimes not for BOX
    void set_centered() ;  
    glm::vec3 halfside() const ; 
    glm::vec3 bmin() const  ; 
    glm::vec3 bmax() const  ; 


    float r1() const ; 
    float z1() const ; 
    float r2() const ; 
    float z2() const ; 
 

    bool is_equal( const nbox& other ) const ; 

    bool is_box   ;   // cannot const these without ctor
    bool is_box3  ; 

};


// only methods that are specific to boxes 
// and need to override the nnode need to be here 


inline NPY_API glm::vec3 nbox::center() const { return glm::vec3(x(), y(), z()) ; }

inline NPY_API float nbox::x() const { return is_box ? param.f.x : 0.f ; }
inline NPY_API float nbox::y() const { return is_box ? param.f.y : 0.f ; }
inline NPY_API float nbox::z() const { return is_box ? param.f.z : 0.f ; }
inline NPY_API bool  nbox::is_centered() const { return x() == 0.f && y() == 0.f && z() == 0.f ; }
inline NPY_API void  nbox::set_centered() 
{ 
    assert( is_box) ; 
    param.f.x = 0.f ; 
    param.f.y = 0.f ; 
    param.f.z = 0.f ; 
}

inline NPY_API glm::vec3 nbox::halfside() const 
{ 
    glm::vec3 h ; 
    if(type == CSG_BOX3)
    { 
        h.x = param.f.x/2.f ;
        h.y = param.f.y/2.f ;
        h.z = param.f.z/2.f ;
    }
    else if(type == CSG_BOX )
    {
        h.x = param.f.w ;
        h.y = param.f.w ;
        h.z = param.f.w ;
    }
    else
    {
        assert(0);
    }
    return h ;
}



inline NPY_API void init_box(nbox* b, const nquad& p )
{
    b->param = p ; 
    b->is_box = true ; 
    b->is_box3 = false ; 
    b->_bbox_model = new nbbox(b->bbox_model()) ;   // bbox_model() has no transforms applied, so is available early

}
inline NPY_API void init_box3(nbox* b, const nquad& p )
{
    b->param = p ; 
    b->is_box = false ; 
    b->is_box3 = true ; 
    b->_bbox_model = new nbbox(b->bbox_model()) ;   // bbox_model() has no transforms applied, so is available early
}

inline NPY_API nbox* make_box(const nquad& p)
{
    nbox* n = new nbox ; 
    nnode::Init(n,CSG_BOX) ; 
    init_box(n, p );
    return n ;
}
inline NPY_API nbox* make_box3(const nquad& p)
{
    nbox* n = new nbox ; 
    nnode::Init(n,CSG_BOX3) ; 
    init_box3(n, p );
    return n ;
}


inline NPY_API nbox* make_box(float x, float y, float z, float w)  // center and halfside
{
    nquad param ;
    param.f =  {x,y,z,w} ;
    return make_box( param ); 
}
inline NPY_API nbox* make_box3(float x, float y, float z, float w=0.f) // three 
{
    assert( w == 0.f );  // used by code gen 
    nquad param ;
    param.f =  {x,y,z,0} ;
    return make_box3( param ); 
}



