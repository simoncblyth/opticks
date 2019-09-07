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

// this code was the starting point for new package YoctoGLRap "YOG"

#include <string>
#include <iostream>
#include <fstream>

#include "YGLTF.h"

using ygltf::glTF_t ; 
using ygltf::node_t ; 
using ygltf::scene_t ; 
using ygltf::mesh_t ; 
using ygltf::mesh_primitive_t ; 
using ygltf::buffer_t ; 
using ygltf::bufferView_t ; 
using ygltf::accessor_t ; 


ygltf::scene_t make_scene(std::vector<int>& nodes)
{
    scene_t sc ;   
    sc.nodes = nodes ;
    return sc ; 
}
ygltf::node_t make_node(int mesh, std::vector<int>& children)
{
    node_t no ; 
    no.mesh = mesh ; 
    no.children = children ; 
    return no ; 
}
ygltf::buffer_t make_buffer(const char* uri,  int byteLength)
{
    buffer_t bu ; 
    bu.uri = uri ; 
    bu.byteLength = byteLength ; 
    return bu ; 
}
ygltf::bufferView_t make_bufferView(
    int buffer, 
    int byteOffset, 
    int byteLength,  
    ygltf::bufferView_t::target_t target
)
{
    bufferView_t bv ; 
    bv.buffer = buffer ; 
    bv.byteOffset = byteOffset ; 
    bv.byteLength = byteLength ; 
    bv.target = target ; 
    return bv ; 
}
ygltf::accessor_t make_accessor(
       int bufferView, 
       int byteOffset, 
       ygltf::accessor_t::componentType_t componentType, 
       int count,
       ygltf::accessor_t::type_t  type,
       std::vector<float>& min, 
       std::vector<float>& max 
)
{
     accessor_t ac ; 
     ac.bufferView = bufferView ; 
     ac.byteOffset = byteOffset ;
     ac.componentType = componentType ; 
     ac.count = count ; 
     ac.type = type ; 
     ac.min = min ; 
     ac.max = max ; 
     return ac ; 
} 

ygltf::mesh_primitive_t make_mesh_primitive(
    std::map<std::string, int>& attributes,
    int indices, 
    int material, 
    ygltf::mesh_primitive_t::mode_t mode
)
{
    mesh_primitive_t mp ; 
    mp.attributes = attributes ; 
    mp.indices = indices ; 
    mp.material = material ; 
    mp.mode = mode ; 
    return mp ; 
}
ygltf::mesh_t make_mesh( std::vector<ygltf::mesh_primitive_t> primitives )
{
    mesh_t mh ; 
    mh.primitives = primitives ; 
    return mh ; 
}

std::unique_ptr<glTF_t> make_gltf_example()
{
    auto gltf = std::unique_ptr<glTF_t>(new glTF_t());

    std::vector<scene_t>& scenes = gltf->scenes ; 
    std::vector<node_t>& nodes = gltf->nodes ; 
    std::vector<buffer_t>& buffers = gltf->buffers ; 
    std::vector<bufferView_t>& bufferViews = gltf->bufferViews ; 
    std::vector<accessor_t>& accessors = gltf->accessors ; 
    std::vector<mesh_t>& meshes = gltf->meshes ; 

    int node = 0 ;   // index of root note 
    std::vector<int> scene_nodes = {node} ;  
    scene_t sc = make_scene(scene_nodes) ;
 

    std::vector<int> children = {} ;

    int mesh = 0 ;  // index of first mesh 
    node_t no = make_node(mesh, children) ; 


    int buffer = 0 ;  // index of first buffer
    buffer_t bu = make_buffer(
         "data:application/octet-stream;base64,AAABAAIAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAACAPwAAAAA=", 
         44
    ); 

    enum {
      indices = 0, 
      vertices = 1,
      num = 2
    };

    bufferView_t bv[num] ; 
    accessor_t   ac[num] ; 

    int indices_byteOffset = 0 ; 
    int indices_byteLength = 6 ; 
    int vertices_byteOffset = 8 ; 
    int vertices_byteLength = 36 ; 

    bv[indices]  = make_bufferView(
          buffer, 
          indices_byteOffset, 
          indices_byteLength,  
          bufferView_t::target_t::element_array_buffer_t 
    ) ; 

    bv[vertices]  = make_bufferView(
          buffer, 
          vertices_byteOffset, 
          vertices_byteLength, 
          bufferView_t::target_t::array_buffer_t 
    ) ; 


    int count = 3 ; 

    std::vector<float> indices_min = { 0 } ; 
    std::vector<float> indices_max = { 2 } ; 

    ac[indices] = make_accessor( 
                          indices, 
                          0,           // byteOffset
                          accessor_t::componentType_t::unsigned_short_t, 
                          count, 
                          accessor_t::type_t::scalar_t,
                          indices_min,
                          indices_max 
                       ) ;   


    std::vector<float> vertices_min = { 0, 0, 0 } ; 
    std::vector<float> vertices_max = { 1, 1, 1 } ; 

    ac[vertices] = make_accessor( 
                          vertices, 
                          0,          // byteOffset
                          accessor_t::componentType_t::float_t, 
                          count, 
                          accessor_t::type_t::vec3_t,
                          vertices_min,
                          vertices_max 
                       ) ;   


    int indices_accessor = indices ; 
    int vertices_accessor = vertices ; 
    int material = -1 ; 
    std::map<std::string, int> attributes = {{"POSITION", vertices_accessor }} ; 

    mesh_primitive_t mp = make_mesh_primitive( 
                             attributes,
                             indices_accessor, 
                             material,
                             mesh_primitive_t::mode_t::triangles_t 
                         ) ; 

    std::vector<mesh_primitive_t> primitives ; 
    primitives = { mp } ;
   
    mesh_t mh = make_mesh( primitives ) ; 

    nodes = {no} ; 
    scenes = {sc} ;  
    buffers = {bu} ; 
    bufferViews = {bv[indices], bv[vertices]} ;  
    accessors = {ac[indices], ac[vertices]} ;
    meshes = {mh} ; 


    return gltf ; 
}


std::unique_ptr<glTF_t> make_gltf()
{
    // NB : there is no checking can easily construct non-sensical gltf 
    //      as shown below 

    auto gltf = std::unique_ptr<glTF_t>(new glTF_t());
    std::vector<node_t>& nodes = gltf->nodes ; 
    std::vector<scene_t>& scenes = gltf->scenes ; 

    node_t a, b, c  ; 
    a.children = {2} ;   
    b.children = {3} ;   
    c.children = {} ;   

    nodes = { a, b, c } ;

    scene_t sc ;    // scene references nodes by index
    sc.nodes = {1,2,3} ;

    scenes = {sc} ;  

    return gltf ; 
}


int main(int argc, char** argv)
{
    //auto gltf = make_gltf() ; 
    auto gltf = make_gltf_example() ; 

    std::string path = "/tmp/UseYoctoGL_Write.gltf" ; 
    bool save_bin = false ; 
    bool save_shaders = false ; 
    bool save_images = false ; 

    save_gltf(path, gltf.get(), save_bin, save_shaders, save_images);

    std::cout << "writing " << path << std::endl ; 

    std::ifstream fp(path);
    std::string line;
    while(std::getline(fp, line)) std::cout << line << std::endl ; 
    return 0 ; 
}
