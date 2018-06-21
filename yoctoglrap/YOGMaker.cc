#include <iostream>
#include <iomanip>

#include "PLOG.hh"
#include "SVec.hh"
#include "BFile.hh"
#include "BStr.hh"
#include "NPY.hpp"
#include "NGLMExt.hpp"

#include "YGLTF.h"

#include "YOG.hh"
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

#include "YOGMakerImpl.hh"

namespace YOG {

void Maker::SaveToGLTF(const NPY<float>* vtx, const NPY<unsigned>* idx, const char* path)
{
    YOG::Sc sc ;
    YOG::Maker mk(&sc) ; 
    YOG::Nd* parent_nd = NULL ;  

    int lvIdx = 0 ; 
    int materialIdx = 0 ; 
    int depth = 0 ; 
    nmat4triple* ltriple = NULL ; 

    int ndIdx = sc.add_node(
                            lvIdx, 
                            materialIdx,
                            "lvName",
                            "pvName",
                            "soName",
                            ltriple,
                            "boundaryName",
                            depth,
                            true,     // selected
                            parent_nd
                            );

     YOG::Mh* mh = sc.get_mesh_for_node( ndIdx );
     mh->vtx = vtx ; 
     mh->idx = idx ; 
 
     mk.convert();
     mk.save(path);
}



void Maker::demo_create(const Geometry& geom)
{
    assert( sc ) ; 

    int vtx_b = add_buffer<float>(geom.vtx, "subfold/subsubfold/vtx.npy");
    int vtx_v = add_bufferView(vtx_b, ARRAY_BUFFER );
    int vtx_a = add_accessor( vtx_v, geom.count,  VEC3,  FLOAT );  
    set_accessor_min_max( vtx_a, geom.vtx_minf, geom.vtx_maxf );

    int idx_b = add_buffer<unsigned>(geom.idx, "subfold/subsubfold/idx.npy");
    int idx_v = add_bufferView(idx_b, ELEMENT_ARRAY_BUFFER );
    int idx_a = add_accessor( idx_v, geom.count,  SCALAR, UNSIGNED_INT );
    set_accessor_min_max( idx_a, geom.idx_minf, geom.idx_maxf );


    int red_mat   = add_material(); 
    set_material_name( red_mat, "red");
    configure_material( red_mat, 1,0,0);  

    int green_mat = add_material(); 
    set_material_name( green_mat, "green");
    configure_material(green_mat, 0,1,0); 

    int blue_mat  = add_material(); 
    set_material_name( blue_mat, "blue");
    configure_material(blue_mat, 0,0,1); 

    int red_mesh = impl->add_mesh();
    int green_mesh = impl->add_mesh();
    int blue_mesh = impl->add_mesh();

    add_primitives_to_mesh( red_mesh, TRIANGLES, vtx_a, idx_a, red_mat );  
    add_primitives_to_mesh( green_mesh, TRIANGLES, vtx_a, idx_a, green_mat );  
    add_primitives_to_mesh( blue_mesh, TRIANGLES, vtx_a, idx_a, blue_mat );  

    //  mesh primitives holds just an index pointer to the material, so 
    //  a single material can be shared by multiple meshes 


    int a = impl->add_node();
    int b = impl->add_node();
    int c = impl->add_node();

    set_node_mesh(a, red_mesh);
    set_node_mesh(b, green_mesh);
    set_node_mesh(c, blue_mesh);
  
    set_node_translation(a,  1.f, 0.f, 0.f); 
    set_node_translation(b,  0.f, 1.f, 0.f); 
    set_node_translation(c,  0.f, 0.f, 1.f); 

    impl->add_scene();

    append_node_to_scene( a );
    //append_node_to_scene( b );
    //append_node_to_scene( c );

    append_child_to_node( b,  a);
    append_child_to_node( c,  a);
}


Maker::Maker(Sc* sc_, bool saveNPYToGLTF_ )  
    :
    sc(sc_),
    impl(new Impl),
    saveNPYToGLTF(saveNPYToGLTF_),
    converted(false)
{
}

void Maker::convert()
{
    assert( converted == false );
    converted = true ; 

    if(impl->num_scene() == 0 )
    {
        impl->add_scene();
        append_node_to_scene( sc->root );
    }

    for(int i=0 ; i < sc->nodes.size() ; i++ )
    {
        Nd* nd = sc->nodes[i] ; 

        int n = impl->add_node();
        node_t& node = impl->get_node(n) ;
 
        node.name = nd->name ;  // pvName 
        node.mesh = nd->soIdx ; 
        node.children = nd->children ; 
        node.extras["boundary"] = nd->boundary ; 
        node.extras["ndIdx"] = nd->ndIdx ; 
        node.extras["parentIdx"] = nd->parent ? nd->parent->ndIdx : -1 ; 
        node.extras["soName"] = nd->mh->soName ; 

        if(nd->transform)
            nglmext::copyTransform( node.matrix, nd->transform->t );
    }

    for(int i=0 ; i < sc->meshes.size() ; i++ )
    {
        Mh* mh = sc->meshes[i] ; 

        int mtIdx = mh->mtIdx ; // material

        int m = impl->add_mesh(); 
        set_mesh_data( m,  mh,  mtIdx ); 

        mesh_t& mesh = impl->get_mesh(m) ;
        mesh.extras["meshIdx"] = m ; 

    }

    for(int i=0 ; i < sc->materials.size() ; i++ )
    {
        Mt* mt = sc->materials[i] ; 
        int m = add_material(); 
        configure_material_auto(m); 
        set_material_name( m, mt->name ); 
    }
}

void Maker::set_mesh_data( int meshIdx, Mh* mh, int materialIdx )
{   
    mesh_t& mesh = impl->get_mesh(meshIdx) ;
    mesh.name = mh->soName ;  // 

    int vtx_a = set_mesh_data_vertices( mh );
    int idx_a = set_mesh_data_indices( mh );
 
    add_primitives_to_mesh( meshIdx, TRIANGLES, vtx_a, idx_a, materialIdx );  
}

std::string Maker::get_mesh_uri( Mh* mh, const char* bufname ) const 
{
    return BFile::FormRelativePath("extras", BStr::itoa(mh->lvIdx), bufname );
}

int Maker::set_mesh_data_vertices( Mh* mh )
{
    const NPY<float>* vtx = mh->vtx ;
    std::string uri = get_mesh_uri( mh, "vertices.npy" );

    std::vector<float> f_min ; 
    std::vector<float> f_max ; 
    vtx->minmax(f_min,f_max);

    SVec<float>::Dump("vtx.fmin", f_min ); 
    SVec<float>::Dump("vtx.fmax", f_max ); 

    int vtx_b = add_buffer<float>(vtx, uri.c_str() );
    int vtx_v = add_bufferView(vtx_b, ARRAY_BUFFER );

    int vtx_a = add_accessor( vtx_v, vtx->getShape(0),  VEC3,  FLOAT );  
    assert( vtx->getNumElements() == 3 );

    set_accessor_min_max( vtx_a, f_min, f_max );

    return vtx_a ; 
}

int Maker::set_mesh_data_indices( Mh* mh )
{
    const NPY<unsigned>* idx = mh->idx ;

    assert( idx->hasShape(-1,1) && " need to idx->reshape(-1,1) first " ); 
    /*
    if(!idx->hasShape(-1,1))
    {
        std::string bef = idx->getShapeString() ; 
        idx->reshape(-1,1); 
        std::string aft = idx->getShapeString() ; 

        LOG(info) 
           << "reshaped idx buffer" 
           << " from " << bef 
           << " to " << aft 
           ; 
    } 
    */

    std::string uri = get_mesh_uri( mh, "indices.npy" );

    std::vector<unsigned> u_min ; 
    std::vector<unsigned> u_max ; 
    idx->minmax(u_min,u_max);

    std::vector<float> f_min(u_min.begin(), u_min.end());
    std::vector<float> f_max(u_max.begin(), u_max.end());

    SVec<float>::Dump("idx.fmin", f_min ); 
    SVec<float>::Dump("idx.fmax", f_max ); 

    int idx_b = add_buffer<unsigned>(idx, uri.c_str() );
    int idx_v = add_bufferView(idx_b, ELEMENT_ARRAY_BUFFER );
    int idx_a = add_accessor( idx_v, idx->getShape(0),  SCALAR, UNSIGNED_INT );
    set_accessor_min_max( idx_a, f_min, f_max );
 
    return idx_a ; 
}

template <typename T> 
int Maker::add_buffer( const NPY<T>* npy, const char* uri )
{
    LOG(info) << "add_buffer" 
              << " uri " << uri 
              << " shape " << npy->getShapeString()
              ;

    specs.push_back(npy->getBufferSpec()) ; 
    NBufferSpec& spec = specs.back() ;
    spec.uri = uri ; 
    assert( spec.ptr == (void*)npy );

    int idx = impl->add_buffer();
    buffer_t& bu = impl->get_buffer(idx)  ; 

    bu.uri = uri ; 
    bu.byteLength = spec.bufferByteLength ; 

    if( saveNPYToGLTF ) // copy data from NPY buffer into glTF buffer
    {
        npy->saveToBuffer( bu.data ) ; 
    }

    return idx ; 
}

int Maker::add_bufferView( int bufferIdx, TargetType_t targetType )
{
    const NBufferSpec& spec = specs[bufferIdx] ;  

    int idx = impl->add_bufferView();
    bufferView_t& bv = impl->get_bufferView(idx) ;  

    bv.buffer = bufferIdx  ;
    bv.byteOffset = spec.headerByteLength ; // offset to skip the NPY header 
    bv.byteLength = spec.dataSize() ;

    switch(targetType)
    {
        case ARRAY_BUFFER          : bv.target = bufferView_t::target_t::array_buffer_t         ; break ;
        case ELEMENT_ARRAY_BUFFER  : bv.target = bufferView_t::target_t::element_array_buffer_t ; break ;
    }
    bv.byteStride = 0 ; 

    return idx ;
}

int Maker::add_accessor( int bufferViewIdx, int count, Type_t type, ComponentType_t componentType ) 
{
    int idx = impl->add_accessor();
    accessor_t& ac = impl->get_accessor(idx) ; 

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
    return idx ;
}

void Maker::set_accessor_min_max(int accessorIdx, const std::vector<float>& minf , const std::vector<float>& maxf )
{
    accessor_t& ac = impl->get_accessor(accessorIdx);
    ac.min = minf ; 
    ac.max = maxf ; 
}

void Maker::append_node_to_scene(int nodeIdx, int sceneIdx)
{
    scene_t& sc = impl->get_scene(sceneIdx);
    sc.nodes.push_back(nodeIdx); 
}

void Maker::append_child_to_node(int childIdx, int nodeIdx ) 
{
    node_t& nd = impl->get_node(nodeIdx);
    nd.children.push_back(childIdx); 
}

void Maker::add_primitives_to_mesh( int meshIdx, Mode_t mode, int positionIdx, int indicesIdx, int materialIdx )
{
    mesh_t& mh = impl->get_mesh(meshIdx) ; 

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
    node_t& node = impl->get_node(nodeIdx) ;  
    node.mesh = meshIdx ;  
}

void Maker::set_node_translation(int nodeIdx, float x, float y, float z)
{
    node_t& node = impl->get_node(nodeIdx) ;  
    node.translation = {{ x, y, z }} ;  
}

int Maker::add_material()
{
    int idx = impl->add_material();
    return idx ; 
}

void Maker::set_material_name(int idx, const std::string& name)
{
    material_t& mt = impl->get_material(idx) ; 
    mt.name = name ;  
}



/**
configure_material_auto
-------------------------

Placeholder for arranging a different appearance 
for each material.

Considered propagating G4 VisAttributes, but 
logvol hold those in G4 model, but in glTF mesh primitives
hold the material index.  Model mismatch ?

**/

void Maker::configure_material_auto( int idx)
{
    switch(idx)
    {
        case 0:  configure_material(idx,  1.0, 0.0, 0.0, 1.0   ,0.5,0.1 ) ;  break ;  
        case 1:  configure_material(idx,  0.0, 1.0, 0.0, 1.0   ,0.5,0.1 ) ;  break ;  
        case 2:  configure_material(idx,  0.0, 0.0, 1.0, 1.0   ,0.5,0.1 ) ;  break ;  
        case 3:  configure_material(idx,  1.0, 1.0, 0.0, 1.0   ,0.5,0.1 ) ;  break ;  
        case 4:  configure_material(idx,  1.0, 0.0, 1.0, 1.0   ,0.5,0.1 ) ;  break ;  
        case 5:  configure_material(idx,  0.0, 1.0, 1.0, 1.0   ,0.5,0.1 ) ;  break ;  
        case 6:  configure_material(idx,  0.5, 0.5, 0.5, 1.0   ,0.5,0.1 ) ;  break ;  
        default: configure_material(idx,  0.9, 0.9, 0.9, 1.0   ,0.5,0.1 ) ;  break ;  
    }
}



void Maker::configure_material(
     int idx, 
     float baseColorFactor_r, 
     float baseColorFactor_g, 
     float baseColorFactor_b, 
     float baseColorFactor_a, 
     float metallicFactor, 
     float roughnessFactor 
    )
{
    material_t& mt = impl->get_material(idx) ; 

    material_pbrMetallicRoughness_t mr ; 
    mr.baseColorFactor =  {{ baseColorFactor_r, baseColorFactor_g, baseColorFactor_b, baseColorFactor_a }} ;
    mr.metallicFactor = metallicFactor ;
    mr.roughnessFactor = roughnessFactor ;

    mt.pbrMetallicRoughness = mr ; 
}

void Maker::save(const char* path, bool cat) const 
{
    assert( converted == true );

    bool createDirs = true ; 
    BFile::preparePath(path, createDirs); 

    bool save_bin = saveNPYToGLTF ; 

    impl->save(path, save_bin); 

    LOG(info) << "writing " << path ; 

    if(cat)
    {
        std::ifstream fp(path);
        std::string line;
        while(std::getline(fp, line)) std::cout << line << std::endl ; 
    } 

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

        std::cout << std::left 
                  << std::setw(3) << i
                  << " "
                  << " spec.uri " << std::setw(40) << spec.uri 
             //     << " bufpath " << bufpath 
                  << std::endl 
                  ;

        if(spec.ptr)
        {
            NPYBase* ptr = const_cast<NPYBase*>(spec.ptr);
            ptr->save(bufpath);
        }
    }
}

template YOG_API int Maker::add_buffer<float>(const NPY<float>*, const char* );
template YOG_API int Maker::add_buffer<unsigned>(const NPY<unsigned>*, const char* );

} // namespace

