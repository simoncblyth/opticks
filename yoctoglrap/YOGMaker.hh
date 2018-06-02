#pragma once

#include <vector>

#include "YGLTF.h"
#include "YOG_API_EXPORT.hh"

class YOG_API YOGMaker 
{
   public:

    static ygltf::scene_t make_scene(std::vector<int>& nodes);
    static ygltf::node_t make_node(int mesh, std::vector<int>& children) ; 

    static ygltf::buffer_t make_buffer(const char* uri,  ygltf::buffer_data_t& data );
    static ygltf::buffer_t make_buffer(const char* uri,  int byteLength);

    static ygltf::bufferView_t make_bufferView(
        int buffer, 
        int byteOffset, 
        int byteLength,  
        ygltf::bufferView_t::target_t target
    );

    static ygltf::accessor_t make_accessor(
           int bufferView, 
           int byteOffset, 
           ygltf::accessor_t::componentType_t componentType, 
           int count,
           ygltf::accessor_t::type_t  type,
           std::vector<float>& min, 
           std::vector<float>& max 
    );

    static ygltf::mesh_primitive_t make_mesh_primitive(
        std::map<std::string, int>& attributes,
        int indices, 
        int material, 
        ygltf::mesh_primitive_t::mode_t mode
    );

    static ygltf::mesh_t make_mesh( std::vector<ygltf::mesh_primitive_t> primitives );

    static std::unique_ptr<ygltf::glTF_t> make_gltf_example();
    static std::unique_ptr<ygltf::glTF_t> make_gltf();

};





