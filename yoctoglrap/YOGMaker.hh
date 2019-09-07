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

#include <string>
#include <vector>
#include "plog/Severity.h"

#include "NBufferSpec.hpp"
#include "YOG_API_EXPORT.hh"

template <typename T> class NPY ;

/**
YOGMaker
==========

Creates renderable glTF by providing the metadata 
to describe the Opticks geocache buffers. 

yzFlip
   adds top node that acts as root and does the flip 
   and points via its children to the "real" root node

saveNPYToGLTF
   enabling this gets ygltf to write the binary buffers, 
   normally it is more convenient to save the buffers 
   separately as ygltf is restricted to existing directories    


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

    static const plog::Severity LEVEL ; 

    static void SaveToGLTF(const NPY<float>* vtx, const NPY<unsigned>* idx, const char* path);

    void demo_create(const Geometry& geom );

    Maker(Sc* sc_=NULL, bool yzFlip_=true, bool saveNPYToGLTF_=false );  
    void convert();

    template <typename T> 
    int add_buffer( const NPY<T>* buffer, const char* uri ); 
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
    void save( const char* path, bool cat=false ) const ; 
    void saveBuffers(const char* path) const ;


    int add_yzflip_top_node(int root );



    Sc*                      sc ;  
    Impl*                    impl ; 
    int                      verbosity ;  

    bool                     yzFlip ;  
    bool                     saveNPYToGLTF ; 
    bool                     converted ; 

    std::vector<NBufferSpec> specs ; 

};

} // namespace


