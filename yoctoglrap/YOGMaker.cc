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

// hmm not pursuing this way, trying a direct translation of sc.py over in YOG.hh

YOGMaker::YOGMaker()  
   :
   m_gltf(new glTF_t()),
   m_scenes(m_gltf->scenes),
   m_nodes(m_gltf->nodes),
   m_buffers(m_gltf->buffers),
   m_bufferViews(m_gltf->bufferViews),
   m_accessors(m_gltf->accessors),
   m_meshes(m_gltf->meshes),
   m_materials(m_gltf->materials)
{
}

std::unique_ptr<glTF_t> YOGMaker::make_gltf(const YOGGeometry& geom )
{
    assert( geom.vtx ) ; 
    assert( geom.vtx_spec ) ; 
    assert( geom.idx ) ; 
    assert( geom.idx_spec ) ; 

    auto gltf = std::unique_ptr<glTF_t>(new glTF_t());

    std::vector<scene_t>& scenes = gltf->scenes ; 
    std::vector<node_t>& nodes = gltf->nodes ; 
    std::vector<buffer_t>& buffers = gltf->buffers ; 
    std::vector<bufferView_t>& bufferViews = gltf->bufferViews ; 
    std::vector<accessor_t>& accessors = gltf->accessors ; 
    std::vector<mesh_t>& meshes = gltf->meshes ; 
    std::vector<material_t>& materials = gltf->materials ; 

    node_t no0 ; 
    no0.mesh = 0 ; 

    node_t no1 ; 
    no1.mesh = 0 ; 
    no1.translation = {{1,0,0}} ; 

    nodes = {no0, no1} ; 
    int no0_ = 0 ; 
    int no1_ = 1 ; 

    scene_t sc ;   
    sc.nodes = {no0_, no1_} ;   
    scenes = {sc} ;  

    material_pbrMetallicRoughness_t mr ; 
    mr.baseColorFactor =  {{ 1.000, 0.766, 0.336, 1.0 }} ;
    mr.metallicFactor = 0.5 ;
    mr.roughnessFactor = 0.1 ;

    material_t mt ; 
    mt.pbrMetallicRoughness = mr ; 

    materials = { mt } ; 
    int mt_ = 0 ; 

    buffer_t vtx_bu ; 
    vtx_bu.uri = "vtx.npy" ; 
    vtx_bu.byteLength = geom.vtx_spec->bufferByteLength ; 
    geom.vtx->saveToBuffer( vtx_bu.data ) ; 

    buffer_t idx_bu ; 
    idx_bu.uri = "idx.npy" ; 
    idx_bu.byteLength = geom.idx_spec->bufferByteLength ; 
    geom.idx->saveToBuffer( idx_bu.data ) ; 

    buffers = {vtx_bu, idx_bu} ; 
    int vtx_bu_ = 0 ; 
    int idx_bu_ = 1 ; 


    // TODO: derive these from npy array content 
    int count = 3 ; 
    std::vector<float> vtx_min = { 0, 0, 0, 1 } ; 
    std::vector<float> vtx_max = { 1, 1, 1, 1 } ; 
    std::vector<float> idx_min = { 0 } ; 
    std::vector<float> idx_max = { 2 } ; 

    bufferView_t vtx_bv ;  
    vtx_bv.buffer = vtx_bu_  ;
    vtx_bv.byteOffset = geom.vtx_spec->headerByteLength ; // offset to skip the NPY header 
    vtx_bv.byteLength = geom.vtx_spec->dataSize() ;
    vtx_bv.target = bufferView_t::target_t::array_buffer_t ;
    vtx_bv.byteStride = 0 ; 
 
    bufferView_t idx_bv ;  
    idx_bv.buffer = idx_bu_  ;
    idx_bv.byteOffset = geom.idx_spec->headerByteLength ; // offset to skip the NPY header
    idx_bv.byteLength = sizeof(unsigned)*count ;  // geom.idx_spec->dataSize() ;
    idx_bv.target = bufferView_t::target_t::element_array_buffer_t ;
    idx_bv.byteStride = 0 ; 
 
    bufferViews = {vtx_bv, idx_bv} ;  
    int vtx_bv_ = 0 ; 
    int idx_bv_ = 1 ; 

    accessor_t vtx_ac ; 
    vtx_ac.bufferView = vtx_bv_ ; 
    vtx_ac.byteOffset = 0 ;
    vtx_ac.componentType = accessor_t::componentType_t::float_t ; 
    vtx_ac.count = count ; 
    vtx_ac.type = accessor_t::type_t::vec4_t ; 
    vtx_ac.min = vtx_min ; 
    vtx_ac.max = vtx_max ; 

    accessor_t idx_ac ; 
    idx_ac.bufferView = idx_bv_ ; 
    idx_ac.byteOffset = 0 ;
    idx_ac.componentType = accessor_t::componentType_t::unsigned_int_t ; 
    idx_ac.count = count ; 
    idx_ac.type = accessor_t::type_t::scalar_t ; 
    idx_ac.min = idx_min ; 
    idx_ac.max = idx_max ; 

    accessors = {vtx_ac, idx_ac} ;
    int vtx_ac_ = 0 ; 
    int idx_ac_ = 1 ; 
 
    mesh_primitive_t mp ; 
    mp.attributes = {{"POSITION", vtx_ac_ }} ; 
    mp.indices = idx_ac_ ; 
    mp.material = mt_ ; 
    mp.mode = mesh_primitive_t::mode_t::triangles_t ; 
 
    mesh_t mh ; 
    mh.primitives = { mp } ; 
    meshes = {mh} ; 

    return gltf ; 
}


