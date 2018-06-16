#include <iostream>
#include <iomanip>

#include "PLOG.hh"
#include "BFile.hh"
#include "NPY.hpp"
#include "YOGMaker.hh"
#include "YOGGeometry.hh"

using ygltf::glTF_t ; 
using ygltf::node_t ; 
using ygltf::scene_t ; 
using ygltf::mesh_t ; 
using ygltf::material_t ; 
using ygltf::material_pbrMetallicRoughness_t ; 
using ygltf::mesh_primitive_t ; 
using ygltf::buffer_t ; 
using ygltf::buffer_data_t ; 
using ygltf::bufferView_t ; 
using ygltf::accessor_t ; 

namespace YOG {

void Maker::demo_create(const Geometry& geom)
{
    int vtx_b = add_buffer<float>(geom.vtx, "subfold/subsubfold/vtx.npy");
    int vtx_v = add_bufferView(vtx_b, ARRAY_BUFFER );
    int vtx_a = add_accessor( vtx_v, geom.count,  VEC4,  FLOAT );  
    set_accessor_min_max( vtx_a, geom.vtx_minf, geom.vtx_maxf );

    int idx_b = add_buffer<unsigned>(geom.idx, "subfold/subsubfold/idx.npy");
    int idx_v = add_bufferView(idx_b, ELEMENT_ARRAY_BUFFER );
    int idx_a = add_accessor( idx_v, geom.count,  SCALAR, UNSIGNED_INT );
    set_accessor_min_max( idx_a, geom.idx_minf, geom.idx_maxf );

    //int mat = add_material(); 
    int mat = add_material(0,1,0); 

    int m = add_mesh();
    add_primitives_to_mesh( m, TRIANGLES, vtx_a, idx_a, mat );  

    int a = add_node();
    int b = add_node();

    set_node_mesh(a, m);
    set_node_mesh(b, m);
  
    set_node_translation(b,  1.f, 0.f, 0.f); 

    add_scene();
    append_node_to_scene( a );
    append_node_to_scene( b );
}


Maker::Maker(bool saveNPYToGLTF_ )  
   :
   gltf(new glTF_t()),
   saveNPYToGLTF(saveNPYToGLTF_)
{
}

template <typename T> 
int Maker::add_buffer( NPY<T>* npy, const char* uri )
{
    specs.push_back(npy->getBufferSpec()) ; 
    NBufferSpec& spec = specs.back() ;
    spec.uri = uri ; 
    assert( spec.ptr == (void*)npy );

    buffer_t bu ; 
    bu.uri = uri ; 
    bu.byteLength = spec.bufferByteLength ; 

    if( saveNPYToGLTF )
    {
        npy->saveToBuffer( bu.data ) ; 
    }

    int buIdx = gltf->buffers.size() ; 
    gltf->buffers.push_back( bu );


    return buIdx ; 
}

int Maker::add_bufferView( int bufferIdx, TargetType_t targetType )
{
    const NBufferSpec& spec = specs[bufferIdx] ;  

    bufferView_t bv ;  
    bv.buffer = bufferIdx  ;
    bv.byteOffset = spec.headerByteLength ; // offset to skip the NPY header 
    bv.byteLength = spec.dataSize() ;

    switch(targetType)
    {
        case ARRAY_BUFFER          : bv.target = bufferView_t::target_t::array_buffer_t         ; break ;
        case ELEMENT_ARRAY_BUFFER  : bv.target = bufferView_t::target_t::element_array_buffer_t ; break ;
    }
    bv.byteStride = 0 ; 

    int bufferViewIdx = gltf->bufferViews.size() ; 
    gltf->bufferViews.push_back( bv );
    return bufferViewIdx ;
}

int Maker::add_accessor( int bufferViewIdx, int count, Type_t type, ComponentType_t componentType ) 
{
    accessor_t ac ; 
    ac.bufferView = bufferViewIdx ; 
    ac.byteOffset = 0 ;
    ac.count = count ; 

    switch(type)
    {
        case SCALAR   : ac.type = accessor_t::type_t::scalar_t ; break ; 
        case VEC2     : ac.type = accessor_t::type_t::vec2_t   ; break ; 
        case VEC3     : ac.type = accessor_t::type_t::vec3_t   ; break ; 
        case VEC4     : ac.type = accessor_t::type_t::vec4_t   ; break ; 
        case MAT2     : ac.type = accessor_t::type_t::mat2_t   ; break ; 
        case MAT3     : ac.type = accessor_t::type_t::mat3_t   ; break ; 
        case MAT4     : ac.type = accessor_t::type_t::mat4_t   ; break ; 
    }

    switch(componentType)
    {
        case BYTE           : ac.componentType = accessor_t::componentType_t::byte_t           ; break ; 
        case UNSIGNED_BYTE  : ac.componentType = accessor_t::componentType_t::unsigned_byte_t  ; break ; 
        case SHORT          : ac.componentType = accessor_t::componentType_t::short_t          ; break ; 
        case UNSIGNED_SHORT : ac.componentType = accessor_t::componentType_t::unsigned_short_t ; break ; 
        case UNSIGNED_INT   : ac.componentType = accessor_t::componentType_t::unsigned_int_t   ; break ; 
        case FLOAT          : ac.componentType = accessor_t::componentType_t::float_t          ; break ; 
    }

    // ac.min and ac.max are skipped

    int accessorIdx = gltf->accessors.size() ; 
    gltf->accessors.push_back( ac );

    return accessorIdx ;
}

void Maker::set_accessor_min_max(int accessorIdx, const std::vector<float>& minf , const std::vector<float>& maxf )
{
    accessor_t& ac = get_accessor(accessorIdx);
    ac.min = minf ; 
    ac.max = maxf ; 
}

int Maker::add_mesh()
{
    int meshIdx = gltf->meshes.size() ; 
    mesh_t mh ; 
    gltf->meshes.push_back( mh );
    return meshIdx ;
}


int Maker::add_scene()
{
    int sceneIdx = gltf->scenes.size() ; 
    scene_t sc ; 
    gltf->scenes.push_back( sc );
    return sceneIdx ;
}

int Maker::add_node()
{
    int nodeIdx = gltf->nodes.size() ; 
    node_t nd ; 
    gltf->nodes.push_back( nd );
    return nodeIdx ;
}

void Maker::append_node_to_scene(int nodeIdx, int sceneIdx)
{
    scene_t& sc = get_scene(sceneIdx);
    sc.nodes.push_back(nodeIdx); 
}

scene_t& Maker::get_scene(int idx)
{
    assert( idx < gltf->scenes.size() );
    return gltf->scenes[idx] ; 
}
mesh_t& Maker::get_mesh(int idx)
{
    assert( idx < gltf->meshes.size() );
    return gltf->meshes[idx] ; 
}
node_t& Maker::get_node(int idx)
{
    assert( idx < gltf->nodes.size() );
    return gltf->nodes[idx] ; 
}
accessor_t& Maker::get_accessor(int idx)
{
    assert( idx < gltf->accessors.size() );
    return gltf->accessors[idx] ; 
}



void Maker::add_primitives_to_mesh( int meshIdx, Mode_t mode, int positionIdx, int indicesIdx, int materialIdx )
{
    assert( meshIdx < gltf->meshes.size() );
    mesh_t& mh = gltf->meshes[meshIdx] ; 

    mesh_primitive_t mp ; 

    mp.attributes = {{"POSITION", positionIdx }} ; 
    mp.indices = indicesIdx  ; 
    mp.material = materialIdx ; 

    switch(mode)
    {
        case POINTS         :  mp.mode = mesh_primitive_t::mode_t::points_t         ; break ; 
        case LINES          :  mp.mode = mesh_primitive_t::mode_t::lines_t          ; break ; 
        case LINE_LOOP      :  mp.mode = mesh_primitive_t::mode_t::line_loop_t      ; break ; 
        case LINE_STRIP     :  mp.mode = mesh_primitive_t::mode_t::line_strip_t     ; break ; 
        case TRIANGLES      :  mp.mode = mesh_primitive_t::mode_t::triangles_t      ; break ; 
        case TRIANGLE_STRIP :  mp.mode = mesh_primitive_t::mode_t::triangle_strip_t ; break ; 
        case TRIANGLE_FAN   :  mp.mode = mesh_primitive_t::mode_t::triangle_fan_t   ; break ; 
    }
 
    mh.primitives.push_back(mp) ;
}

void Maker::set_node_mesh(int nodeIdx, int meshIdx)
{
    node_t& node = get_node(nodeIdx) ;  
    node.mesh = meshIdx ;  
}

void Maker::set_node_translation(int nodeIdx, float x, float y, float z)
{
    node_t& node = get_node(nodeIdx) ;  
    node.translation = {{ x, y, z }} ;  
}

int Maker::add_material(
     float baseColorFactor_r, 
     float baseColorFactor_g, 
     float baseColorFactor_b, 
     float baseColorFactor_a, 
     float metallicFactor, 
     float roughnessFactor 
    )
{
    material_pbrMetallicRoughness_t mr ; 
    mr.baseColorFactor =  {{ baseColorFactor_r, baseColorFactor_g, baseColorFactor_b, baseColorFactor_a }} ;
    mr.metallicFactor = metallicFactor ;
    mr.roughnessFactor = roughnessFactor ;

    material_t mt ; 
    mt.pbrMetallicRoughness = mr ; 

    int materialIdx = gltf->materials.size() ; 
    gltf->materials.push_back( mt );
    return materialIdx ;
}

void Maker::save(const char* path) const 
{
    bool save_bin = saveNPYToGLTF ; 
    bool save_shaders = false ; 
    bool save_images = false ; 

    bool createDirs = true ; 
    BFile::preparePath(path, createDirs); 

    save_gltf(path, gltf, save_bin, save_shaders, save_images);
    std::cout << "writing " << path << std::endl ; 

    std::ifstream fp(path);
    std::string line;
    while(std::getline(fp, line)) std::cout << line << std::endl ; 

    if(!save_bin)
        saveBuffers(path);
}

void Maker::saveBuffers(const char* path) const 
{
    // This can save buffers into non-existing subfolders, 
    // which it creates, unlike the ygltf buffer saving.

    LOG(info) << " path " << path ; 

    std::string dir = BFile::ParentDir(path);

    for(unsigned i=0 ; i < specs.size() ; i++)
    {
        const NBufferSpec& spec = specs[i]; 

        std::string bufpath_ = BFile::FormPath( dir.c_str(),  spec.uri.c_str() ); 
        const char* bufpath = bufpath_.c_str() ; 

        std::cout << std::setw(3) << i
                  << " "
                  << " spec.uri " << spec.uri 
                  << " bufpath " << bufpath 
                  << std::endl 
                  ;

        if(spec.ptr)
        {
            NPYBase* ptr = const_cast<NPYBase*>(spec.ptr);
            ptr->save(bufpath);
        }
    }
}

template YOG_API int Maker::add_buffer<float>(NPY<float>*, const char* );
template YOG_API int Maker::add_buffer<unsigned>(NPY<unsigned>*, const char* );


} // namespace



