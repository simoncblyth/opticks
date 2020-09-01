/**
UseOptiXGeometryInstanced
==========================

Start from UseOptiXGeometryInstancedStandalone, plan:

1. DONE: adopt Opticks packages to reduce the amount of code
2. DONE: adopt OCtx watertight wrapper, adding whats needed for instancing  
3. DONE: add optional switch from box to sphere 
4. get a layered texture to work with instances, such that 
   different groups of instances use different layers of the texture 
5. DONE: generate PPM of thousands of textured Earths with some visible variation 
6. layered 1d float texture

* the OCtx branch is mostly working, bit the traditional way is 
  suffering from very dark renders when texturing 

Next onto UseOptiXGeometryInstancedOCtx starting with the OCtx branch of this, 
as its too difficult to do new things in two ways at once.

**/

#include <chrono>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <fstream>

#include "OKConf.hh"
#include "SStr.hh"
#include "BFile.hh"
#include "OPTICKS_LOG.hh"
#include "OXPPNS.hh"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "ImageNPY.hpp"

#define USE_OCTX 1 

#ifdef USE_OCTX
#include "OCtx.hh"
#endif
#include "OTex.hh"   // hmm uses OCtx internally 
#include "OBuffer.hh"


#include "CMAKE_TARGET.hh" 


glm::mat4 MakeTransform(unsigned u, unsigned v, unsigned w)
{
    //glm::vec4 rot( rand(), rand(), rand(),  rand()*360.f ); // random axis and angle 
    //glm::vec4 rot(  0,  0, 1,  rand()*360.f );
    glm::vec4 rot(  0,  0, 1,  0 );
    glm::vec3 sca( 0.5 ) ; 
    glm::vec3 tla(  10.f*u , 10.f*v , -10.f*w ) ; 
    glm::mat4 m4 = nglmext::make_transform("trs", tla, rot, sca );
    return m4 ; 
}

NPY<float>* MakeTransforms(unsigned nu, unsigned nv, unsigned nw)
{
    unsigned num_tr = nu*nv*nw ; 
    NPY<float>* tr = NPY<float>::make(num_tr, 4, 4); 
    tr->zero(); 
    bool transpose = false ; 
    unsigned count(0) ; 
    for( unsigned u = 0; u < nu; ++u ) { 
    for( unsigned v = 0; v < nv ; ++v ) { 
    for( unsigned w = 0; w < nw ; ++w ) { 

        glm::mat4 m4 = MakeTransform(u,v,w); 
        tr->setMat4(m4, count, -1, transpose); 
        count++ ; 
    }
    }
    }
    assert( count == num_tr ); 
    return tr ; 
}


void MakeGeometry(const unsigned nu, const unsigned nv, const unsigned nw, const char* main_ptx, unsigned entry_point_index, bool single, const char* closest_hit)
{
    //  with (nu,nv,nw) of (100,100,4) have observed some flakiness that manifests as "corrupted" looking texture
    // initially interpreted this as a problem with layered textures

    NPY<float>* transforms = MakeTransforms(nu,nv,nw); 
    std::string transforms_digest = transforms->getDigestString(); 
    unsigned num_transforms = transforms->getNumItems(); 
    LOG(info) << "MakeTransforms num_transforms " << num_transforms << " transforms_digest " << transforms_digest ; 
    assert( num_transforms == nu*nv*nw );  

    NPY<float>* transforms0 = NPY<float>::make_modulo_selection(transforms, 2, 0 );
    NPY<float>* transforms1 = NPY<float>::make_modulo_selection(transforms, 2, 1 );

    LOG(info) << " transforms0 " << transforms0->getShapeString(); 
    LOG(info) << " transforms1 " << transforms1->getShapeString(); 

    std::string transforms_digest2 = transforms->getDigestString(); 
    LOG(info) << "MakeTransforms num_transforms " << num_transforms << " transforms_digest2 " << transforms_digest2 ; 
    assert( strcmp( transforms_digest2.c_str(), transforms_digest.c_str() ) == 0 ); 
 
    float sz = 10.f ; 
    unsigned nbox = 10u ; 

    const char* rubox_ptx = OKConf::PTXPath( CMAKE_TARGET, "rubox.cu" ) ; 
    const char* sphere_ptx = OKConf::PTXPath( CMAKE_TARGET, "sphere.cu" ) ; 

    void* box_ptr = OCtx::Get()->create_geometry(nbox, rubox_ptx, "rubox_bounds", "rubox_intersect" ); 
    OCtx::Get()->set_geometry_float3(box_ptr, "boxmin", -sz/2.f, -sz/2.f, -sz/2.f ); 
    OCtx::Get()->set_geometry_float3(box_ptr, "boxmax",  sz/2.f,  sz/2.f,  sz/2.f ); 

    void* sph_ptr = OCtx::Get()->create_geometry(nbox, sphere_ptx, "bounds", "intersect" ); 
    OCtx::Get()->set_geometry_float4( sph_ptr, "sphere",  0, 0, 0, 10.0 );

    void* mat_ptr = OCtx::Get()->create_material( main_ptx,  closest_hit, entry_point_index ); 

    void* top_ptr = OCtx::Get()->create_group("top_object", NULL );  
    void* top_accel = OCtx::Get()->create_acceleration("Trbvh");
    OCtx::Get()->set_group_acceleration( top_ptr, top_accel ); 

    if(single)
    {
        glm::mat4 m4box = MakeTransform(1,1,0); 
        glm::mat4 m4sph = MakeTransform(1,1,1); 
        void* box_assembly_ptr = OCtx::Get()->create_single_assembly( m4box, box_ptr, mat_ptr );
        void* sph_assembly_ptr = OCtx::Get()->create_single_assembly( m4sph, sph_ptr, mat_ptr );

        OCtx::Get()->group_add_child_group( top_ptr, box_assembly_ptr ); 
        OCtx::Get()->group_add_child_group( top_ptr, sph_assembly_ptr ); 
    }
    else
    {
        void* box_assembly_ptr = OCtx::Get()->create_instanced_assembly( transforms0, box_ptr, mat_ptr );
        void* sph_assembly_ptr = OCtx::Get()->create_instanced_assembly( transforms1, sph_ptr, mat_ptr );

        OCtx::Get()->group_add_child_group( top_ptr, box_assembly_ptr ); 
        OCtx::Get()->group_add_child_group( top_ptr, sph_assembly_ptr ); 
    }
}


void SetupView(unsigned width, unsigned height, unsigned nu, unsigned nv, unsigned nw)
{
    float scene_epsilon = 1.f ;  // when cut into earth, see circle of back to front geography from inside the sphere

    float extent = 100.0 ; 
    glm::vec4 ce_m(float(nu),float(nv), 0.f, extent ); 
    //glm::vec3 eye_m(  0.f, 0.f, 0.1f  );  //  viewpoint in unit model frame 
    glm::vec3 eye_m( -0.1, -0.1f, 0.1f  );  //  viewpoint in unit model frame 
    glm::vec3 look_m( 0.7f, 0.7f, -0.7); 
    glm::vec3 up_m(   0.f, 0.f, 1.f   ); 

    float tanYfov = 1.f ; 
    const bool dump = false ;
    glm::vec3 eye,U,V,W ;
    nglmext::GetEyeUVW( ce_m, eye_m, look_m, up_m, width, height, tanYfov, eye, U, V, W, dump );

    OCtx::Get()->set_context_viewpoint( eye, U, V, W, scene_epsilon );
}


/**
SetupTextures
-------------

Loads a single PPM path several times with different modifications like add_border, add_midline, add_quadline 
which each yield 2d textures which are uploaded to GPU.

**/
void SetupTextures(const char* path, unsigned& tex_width, unsigned& tex_height, std::vector<int>& tex_id)
{
    const bool yflip0 = false ; 
    unsigned ncomp = 4 ; 
    const unsigned NIMG = 3 ; 
    tex_id.resize(NIMG); 
    std::fill(tex_id.begin(), tex_id.end(), -1);
 
    typedef enum { CONCAT, SPLIT, ONE } Mode_t ; 
    Mode_t mode = CONCAT ; 

    const char* config[NIMG] ; 
    config[0] = "add_border" ; 
    config[1] = "add_midline" ; 
    config[2] = "add_quadline" ; 

    const char* param_key[NIMG] ; 
    param_key[0] = "tex_param_0" ; 
    param_key[1] = "tex_param_1" ; 
    param_key[2] = "tex_param_2" ; 

    NPY<unsigned char>* inpc = NULL ; 
    NPY<unsigned char>* inp[NIMG] ;  
 
    if( mode == SPLIT )
    {
        LOG(info) << " SPLIT : separate loading of PPM into distinct arrays " ;  
        for(unsigned i=0 ; i < NIMG ; i++)
        {
            bool concat_dimension = false ; 
            inp[i] = ImageNPY::LoadPPM( path, yflip0, ncomp, config[i], concat_dimension ); 
            tex_id[i] = OCtx::Get()->upload_2d_texture(param_key[i], inp[i], "INDEX_NORMALIZED_COORDINATES", -1 );  
            LOG(info) 
                << " i " << i 
                << " inp " << inp[i]->getShapeString() 
                << " tex_id " << tex_id[i] 
                ; 
        }
        tex_height = inp[0]->getShape(0);    // assumes all img are same size
        tex_width  = inp[0]->getShape(1);  
    } 
    else if ( mode == CONCAT ) 
    {
        LOG(info) << "[ CONCAT : collective array holding multiple images " ;  
        std::vector<std::string> paths = {path, path, path} ;
        std::vector<std::string> configs = {config[0], config[1], config[2] };

        bool old_concat = true ;   // should make no difference, now that have fixed the old imp
        inpc = ImageNPY::LoadPPMConcat(paths, configs, yflip0, ncomp, old_concat ) ; 
        //ImageNPY::SavePPMConcat(inpc, path, yflip0);  // just for debug

        for(unsigned i=0 ; i < NIMG ; i++)
        {
            tex_id[i] = OCtx::Get()->upload_2d_texture(param_key[i], inpc, "INDEX_NORMALIZED_COORDINATES", i );  
            LOG(info)
                << " i " << i 
                << " param_key " << param_key[i]
                << " tex_id " << tex_id[i]
                ;
        }
        tex_height = inpc->getShape(1);  
        tex_width  = inpc->getShape(2);  
        LOG(info) << "] CONCAT " ;  
    }
    else if ( mode == ONE )
    {
        LOG(info) << " ONE " ;  
        bool concat_dimension = true ; 
        inpc = ImageNPY::LoadPPM(path, yflip0, ncomp, config[0], concat_dimension) ; 
        tex_id[0] = OCtx::Get()->upload_2d_texture(param_key[0], inpc, "INDEX_NORMALIZED_COORDINATES", -1 );  
        tex_id[1] = -1 ; 
        tex_id[2] = -1 ; 

        tex_height = inpc->getShape(1);  
        tex_width  = inpc->getShape(2);  
    }
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info) << "CMAKE_TARGET " << CMAKE_TARGET ; 
    assert( strcmp(CMAKE_TARGET, "UseOptiXGeometryInstancedOCtx") == 0 ) ;

    const char* cu_name = SStr::Concat(CMAKE_TARGET, ".cu" ) ;   
    const char* main_ptx = OKConf::PTXPath( CMAKE_TARGET, cu_name );  
    const char* tmpdir = SStr::Concat("$TMP/", CMAKE_TARGET ) ; 

    const char* path_default = "/tmp/SPPMTest_MakeTestImage.ppm" ; 
    const char* path = argc > 1 ? argv[1] : path_default ; 

    //const char* config_default = "tex0,textest" ; 
    const char* config_default = "tex0" ; 
    const char* config = argc > 2 ? argv[2] : config_default ; 

    int tex_index_default = 0 ;  
    int tex_index = tex_index_default ; 
    if(SStr::Contains(config, "tex0")) tex_index = 0 ; 
    if(SStr::Contains(config, "tex1")) tex_index = 1 ; 
    if(SStr::Contains(config, "tex2")) tex_index = 2 ; 

    bool with_geometry = !SStr::Contains(config, "textest" );  
    bool single = SStr::Contains(config, "single" ); 

    const char* closest_hit_default = "closest_hit_normal"  ; 
    const char* closest_hit = closest_hit_default ; 
    if(SStr::Contains(config,"normal")) closest_hit = "closest_hit_normal" ; 
    if(SStr::Contains(config,"local"))  closest_hit = "closest_hit_local" ; 
    if(SStr::Contains(config,"global")) closest_hit = "closest_hit_global" ; 
    if(SStr::Contains(config,"textured")) closest_hit = "closest_hit_textured" ; 

    LOG(info) 
        << " args "
        << " path " << path  
        << " config " << config
        << " tex_index " << tex_index
        << " with_geometry " << with_geometry 
        << " single " << single 
        << " closest_hit " << closest_hit  
        ;  

    unsigned tex_width ; 
    unsigned tex_height ;
    std::vector<int> tex_id ; 
    SetupTextures(path, tex_width, tex_height, tex_id ); 

    assert( tex_index < int(tex_id.size()) ); 
    int texture_id = tex_id[tex_index] ; 
    LOG(info) 
        << " tex_index " << tex_index
        << " texture_id " << texture_id
        << " tex_width " << tex_width
        << " tex_height " << tex_height
        ; 
    assert( texture_id > 0 ); 
    OCtx::Get()->set_context_int("texture_id", texture_id); 

    unsigned width = 1024 ; 
    unsigned height = 512 ; 

    const unsigned entry_point_index = 0u ;
    const char* raygen = NULL ; 

    if( with_geometry )
    {
        const unsigned nu = 10u;
        const unsigned nv = 10u;
        const unsigned nw = 4u;
        MakeGeometry( nu, nv, nw, main_ptx, entry_point_index, single, closest_hit ); 
        SetupView(width, height, nu, nv, nw);  
        raygen = "raygen" ; 
    } 
    else
    {
        height = tex_height ; 
        width =  tex_width ; 
        raygen = "raygen_texture_test" ; 
        LOG(info) << " height " << height << " width " << width ; 
    }

    OCtx::Get()->set_raygen_program( entry_point_index, main_ptx, raygen );
    OCtx::Get()->set_miss_program(   entry_point_index, main_ptx, "miss" );

    NPY<unsigned char>* out = NPY<unsigned char>::make(height, width, 4);
    void* outBuf = OCtx::Get()->create_buffer(out, "output_buffer", 'O', ' ', -1);
    OCtx::Get()->desc_buffer( outBuf );

    double t_prelaunch ; 
    double t_launch ; 
    OCtx::Get()->launch_instrumented( entry_point_index, width, height, t_prelaunch, t_launch );

    out->zero();  
    OCtx::Get()->download_buffer(out, "output_buffer", -1);
    const bool yflip = true ;
    ImageNPY::SavePPM(tmpdir, "out.ppm", out, yflip );

    return 0 ; 
}


