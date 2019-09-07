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

#include <sstream>

#include "PLOG.hh"
#include "NCSG.hpp"
#include "NTriSource.hpp"
#include "NCSGBSP.hpp"

// ocsgbsp- external 
#include "csgjs.hh"    

csgjs_csgnode* NCSGBSP::ConvertToBSP( const NTriSource* tris)
{
    csgjs_model model ;

    unsigned nvtx = tris->get_num_vert();
    unsigned ntri = tris->get_num_tri();

    glm::uvec3 tri ; 
    glm::vec3 pos ; 
    glm::vec3 nrm ; 
    glm::vec3 uv ;
 
    csgjs_vertex vtx ; 

    for(unsigned i=0 ; i < ntri ; i++)
    {
        tris->get_tri(i, tri );

        model.indices.push_back(tri.x);
        model.indices.push_back(tri.y);
        model.indices.push_back(tri.z);
    }

    for(unsigned i=0 ; i < nvtx ; i++)
    {
        tris->get_vert(i, pos );
        tris->get_normal(i, nrm );
        tris->get_uv(i, uv );

        vtx.pos = csgjs_vector(pos.x, pos.y, pos.z);
        vtx.normal = csgjs_vector(nrm.x, nrm.y, nrm.z);
        vtx.uv = csgjs_vector(uv.x, uv.y, uv.z) ;

        model.vertices.push_back(vtx) ;
    }

    csgjs_csgnode* bsp_node = new csgjs_csgnode(csgjs_modelToPolygons(model));
    return bsp_node ; 
}


NCSGBSP::NCSGBSP(const NTriSource* left_, const NTriSource* right_, OpticksCSG_t operation )
    :
    left(ConvertToBSP(left_)),
    right(ConvertToBSP(right_)),
    operation(operation),
    combined(NULL),
    model(NULL)
{
    init();
}

void NCSGBSP::init()
{
    switch(operation)
    {
        case CSG_UNION:        combined = csg_union(left, right)     ; break ; 
        case CSG_INTERSECTION: combined = csg_intersect(left, right) ; break ; 
        case CSG_DIFFERENCE  : combined = csg_subtract(left, right)  ; break ; 
        default: assert(0)  ;
    }
    assert(combined);

    std::vector<csgjs_polygon> polygons = combined->allPolygons();

    model = new csgjs_model ;

    int p = 0;
    for (size_t i = 0; i < polygons.size(); i++)
    {
        const csgjs_polygon & poly = polygons[i];
        unsigned nv = poly.vertices.size() ; 
        assert( nv == 3 );

        for (size_t j = 2; j < nv ; j++)
        {
            model->vertices.push_back(poly.vertices[0]);     model->indices.push_back(p++);
            model->vertices.push_back(poly.vertices[j - 1]); model->indices.push_back(p++);
            model->vertices.push_back(poly.vertices[j]);     model->indices.push_back(p++);
        }
    }

    LOG(info) << "NCSGBSP::init"
              << " npoly " << polygons.size()
              ;

}



unsigned NCSGBSP::get_num_tri() const 
{
    unsigned n_indices = model->indices.size() ;
    assert( n_indices % 3 == 0);
    return n_indices/3 ; 
}
unsigned NCSGBSP::get_num_vert() const 
{
    return model->vertices.size() ;
}
void NCSGBSP::get_vert( unsigned i, glm::vec3& v ) const 
{
    const csgjs_vertex& vtx = model->vertices[i] ; 
    v.x = vtx.pos.x ; 
    v.y = vtx.pos.y ; 
    v.z = vtx.pos.z ; 
}
void NCSGBSP::get_normal( unsigned i, glm::vec3& n ) const 
{
    const csgjs_vertex& vtx = model->vertices[i] ; 
    n.x = vtx.normal.x ; 
    n.y = vtx.normal.y ; 
    n.z = vtx.normal.z ; 
}
void NCSGBSP::get_uv( unsigned i, glm::vec3& uv ) const 
{
    const csgjs_vertex& vtx = model->vertices[i] ; 
    uv.x = vtx.uv.x ; 
    uv.y = vtx.uv.y ; 
    uv.z = vtx.uv.z ; 
}
void NCSGBSP::get_tri(unsigned i, glm::uvec3& t) const 
{
    const std::vector<int>& indices = model->indices ; 
    t.x = indices[i*3+0] ;
    t.y = indices[i*3+1] ;
    t.z = indices[i*3+2] ;
}
void NCSGBSP::get_tri(unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const 
{
    get_tri(i, t );
    get_vert(t.x, a );
    get_vert(t.y, b );
    get_vert(t.z, c );
}



