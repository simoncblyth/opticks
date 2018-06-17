#pragma once

#include <string>
#include <vector>

#include "NBufferSpec.hpp"
#include "YOG_API_EXPORT.hh"

template <typename T> class NPY ;

/**
YOGMaker
==========

* creates renderable glTF by providing the metadata 
  to describe the Opticks geocache buffers. 



Dependencies
-------------

Note that there is no dependency on the glTF implementation 
in this header, that is hidden inside YOGMakerImpl.hh and YOGMaker.cc 

The motivation for this arrangement is to isolate users of YOG::Maker
such as X4PhysicalVolume from the specific glTF implementation in use.



**/

namespace YOG {

struct Sc ; 
struct Mh ; 
struct Geometry ;

struct Impl ; 

typedef enum { SCALAR, VEC2, VEC3, VEC4, MAT2, MAT3, MAT4 } Type_t ; 
typedef enum { BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, UNSIGNED_INT, FLOAT} ComponentType_t ;
typedef enum { ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER } TargetType_t ;
typedef enum { POINTS, LINES, LINE_LOOP, LINE_STRIP, TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN } Mode_t ; 

struct YOG_API Maker 
{
    void demo_create(const Geometry& geom );

    Maker(Sc* sc_=NULL, bool saveNPYToGLTF_=false );  
    void convert();

    template <typename T> 
    int add_buffer( NPY<T>* buffer, const char* uri ); 
    int add_bufferView( int bufferIdx, TargetType_t targetType ); 
    int add_accessor( int bufferViewIdx, int count, Type_t type, ComponentType_t componentType ) ;
    void set_accessor_min_max(int accessorIdx, const std::vector<float>& minf , const std::vector<float>& maxf );

    int add_material();
    void set_material_name(int idx, const std::string& name);
    void configure_material_auto( int idx);
    void configure_material(
        int idx, 
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
    void append_child_to_node(int childIdx, int nodeIdx ) ; 

    int add_mesh() ;
    void set_mesh_data( int meshIdx, Mh* mh, int materialIdx );
    int  set_mesh_data_indices( Mh* mh );
    int  set_mesh_data_vertices( Mh* mh );
    std::string get_mesh_uri( Mh* mh, const char* bufname ) const ;


    int add_node() ;
    void set_node_mesh(int nodeIdx, int meshIdx);
    void set_node_translation(int nodeIdx, float x, float y, float z);
    void save( const char* path) const ; 
    void saveBuffers(const char* path) const ;


    Sc*                      sc ;  
    Impl*                    impl ; 

    bool                     saveNPYToGLTF ; 
    bool                     converted ; 
    std::vector<NBufferSpec> specs ; 

};

} // namespace


