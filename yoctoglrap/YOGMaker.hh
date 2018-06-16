#pragma once

#include <vector>

#include "YGLTF.h"
#include "NBufferSpec.hpp"
#include "YOG_API_EXPORT.hh"

namespace ygltf 
{
    struct glTF_t ;   
    struct node_t ;
    struct mesh_t ;
}

template <typename T> class NPY ;

/**
YOGMaker
==========

This focusses on attempting to create renderable GLTF 
by providing the metadata to describe the buffers. 

**/

namespace YOG {

struct Geometry ;

typedef enum { SCALAR, VEC2, VEC3, VEC4, MAT2, MAT3, MAT4 } Type_t ; 
typedef enum { BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, UNSIGNED_INT, FLOAT} ComponentType_t ;
typedef enum { ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER } TargetType_t ;
typedef enum { POINTS, LINES, LINE_LOOP, LINE_STRIP, TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN } Mode_t ; 

struct YOG_API Maker 
{
    void demo_create(const Geometry& geom );

    Maker(bool saveNPYToGLTF_=false);

    template <typename T> 
    int add_buffer( NPY<T>* buffer, const char* uri ); 

    int add_bufferView( int bufferIdx, TargetType_t targetType ); 
    int add_accessor( int bufferViewIdx, int count, Type_t type, ComponentType_t componentType ) ;
    void set_accessor_min_max(int accessorIdx, const std::vector<float>& minf , const std::vector<float>& maxf );

    int add_material(
        float baseColorFactor_r=1.000, 
        float baseColorFactor_g=0.766, 
        float baseColorFactor_b=0.336, 
        float baseColorFactor_a=1.0, 
        float metallicFactor=0.5, 
        float roughnessFactor=0.1 
        );
     
    void add_primitives_to_mesh( int meshIdx, Mode_t mode, int positionIdx, int indicesIdx, int materialIdx );
    int add_scene() ;
    void append_node_to_scene(int nodeIdx, int sceneIdx=0) ;  // roots 

    int add_mesh() ;
    int add_node() ;
    void set_node_mesh(int nodeIdx, int meshIdx);
    void set_node_translation(int nodeIdx, float x, float y, float z);

    ygltf::scene_t& get_scene(int idx=0);
    ygltf::mesh_t& get_mesh(int idx);
    ygltf::node_t& get_node(int idx);
    ygltf::accessor_t& get_accessor(int idx);

    void save( const char* path) const ; 
    void saveBuffers(const char* path) const ;

    ygltf::glTF_t*  gltf ; 
    bool saveNPYToGLTF ; 
    std::vector<NBufferSpec> specs ; 

};

} // namespace


