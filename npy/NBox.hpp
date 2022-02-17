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

#. CSG_BOX3 (symmetrically placed at origin with fullside param)   <-- THIS IS THE STANDARD ONE 
#. CSG_BOX (positioned box with single dimension control) 
#. TODO: CSG_BOX6 (box with bmin bmax param) for nudge convenience

*/

struct nmat4triple ; 

struct NPY_API nbox : nnode 
{
    static nbox* Create(const nquad& p,                     OpticksCSG_t type  ); 
    static nbox* Create(float x, float y, float z, float w, OpticksCSG_t type  );

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

    static float sdf_local_(const glm::vec3& pos, const glm::vec3& halfside); 

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

    bool is_box() const  ;  
    bool is_box3() const  ; 

};




