#pragma once

#include <cassert>
#include "YGLTF.h"

using ygltf::glTF_t ; 
using ygltf::node_t ; 
using ygltf::scene_t ; 
using ygltf::mesh_t ; 
using ygltf::material_t ; 
using ygltf::buffer_t ; 
using ygltf::bufferView_t ; 
using ygltf::accessor_t ; 

namespace YOG {

/**
YOG::Impl
=========

Notice no YOG_API, these symbols are not exported.
But as everything is in the header, isnt that mute ?

* YES : but if a user imports YOGMakerImpl (eg tests/YOGMakerMonolithicTest.cc) 
  then they declare that they are wish to depend on internals.

**/

struct Impl {

    glTF_t* gltf ; 

    Impl() : gltf(new glTF_t()) {}


    unsigned num_scene() const 
    { 
        return gltf->scenes.size() ;
    }
    unsigned num_material() const 
    { 
        return gltf->materials.size() ;
    }

    scene_t& get_scene(int idx)
    {
        assert( idx < gltf->scenes.size() );
        return gltf->scenes[idx] ; 
    }
    node_t& get_node(int idx)
    {
        assert( idx < gltf->nodes.size() );
        return gltf->nodes[idx] ; 
    }
    mesh_t& get_mesh(int idx)
    {
        assert( idx < gltf->meshes.size() );
        return gltf->meshes[idx] ; 
    }
    buffer_t& get_buffer(int idx)
    {
        assert( idx < gltf->buffers.size() );
        return gltf->buffers[idx] ; 
    }
    bufferView_t& get_bufferView(int idx)
    {
        assert( idx < gltf->bufferViews.size() );
        return gltf->bufferViews[idx] ; 
    }
    accessor_t& get_accessor(int idx)
    {
        assert( idx < gltf->accessors.size() );
        return gltf->accessors[idx] ; 
    }
    material_t& get_material(int idx)
    {
        assert( idx < gltf->materials.size() );
        return gltf->materials[idx] ; 
    }

    int add_scene()
    {
        int idx = gltf->scenes.size() ; 
        scene_t obj ; 
        gltf->scenes.push_back( obj );
        return idx ;
    }
    int add_node()
    {
        int idx = gltf->nodes.size() ; 
        node_t obj ; 
        gltf->nodes.push_back( obj );
        return idx ;
    }
    int add_mesh()
    {
        int idx = gltf->meshes.size() ; 
        mesh_t obj ; 
        gltf->meshes.push_back( obj );
        return idx ;
    }
    int add_buffer()
    {
        int idx = gltf->buffers.size() ; 
        buffer_t obj ; 
        gltf->buffers.push_back( obj );
        return idx ;
    }
    int add_bufferView()
    {
        int idx = gltf->bufferViews.size() ; 
        bufferView_t obj ; 
        gltf->bufferViews.push_back( obj );
        return idx ;
    }
    int add_accessor()
    {
        int idx = gltf->accessors.size() ; 
        accessor_t obj ; 
        gltf->accessors.push_back( obj );
        return idx ;
    }
    int add_material()
    {
        int idx = gltf->materials.size() ; 
        material_t obj ; 
        gltf->materials.push_back( obj );
        return idx ;
    }

    void save(const char* path, bool save_bin)
    {
        bool save_shaders = false ; 
        bool save_images = false ; 
        save_gltf(path, gltf, save_bin, save_shaders, save_images);
    }

};

}  // namespace




