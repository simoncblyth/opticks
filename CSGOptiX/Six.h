#pragma once
/**
Six : backward compatibility layer :  enabling new workflow CSGFoundry geometry to be used with OptiX < 7 
===========================================================================================================

**/

#include <map>
#include <vector>
#include <optixu/optixpp_namespace.h>

struct qat4 ; 

//class Opticks ; 

struct CSGFoundry ; 
struct CSGSolid ; 

struct Params ; 
struct Frame ; 

struct Six
{
    //const Opticks*          ok ; 
    //const std::vector<unsigned>&  solid_selection ; 

    std::vector<unsigned>   solid_selection ; 
    unsigned long long      emm ; 

    optix::Context     context ;
    optix::Material    material ;

    // note that output buffers cannot with replaced with plain CUDA buffers
    // so must create these optix::Buffer and then grab their CUDA pointers 
    optix::Buffer      pixel_buffer ;    // uchar4
    optix::Buffer      isect_buffer ;    // float4
    optix::Buffer      photon_buffer ;   // quad4

    uchar4* d_pixel ; 
    float4* d_isect ; 
    quad4*  d_photon ; 

    Params*           params ; 
    const char*       ptx_path ; 
    const char*       geo_ptx_path ; 
    unsigned          entry_point_index ; 
    unsigned          optix_device_ordinal ; 
    const CSGFoundry* foundry ; 
    int               pindex ; 

    std::map<unsigned, optix::Geometry> solids ; 

    std::vector<optix::Group>    groups ; 

    Six(const char* ptx_path, const char* geo_ptx_path, Params* params_);  
    void initContext();
    void initFrame(); 

    void updateContext();  // for changed params such as viewpoint 
    void initPipeline();
    void setFoundry(const CSGFoundry* foundry);

    void createGeom();
    void createContextInputBuffers();
    void createGAS();
    void createGAS_Standard();
    void createGAS_Selection();
    void createIAS();
    void createIAS_Standard();
    void createIAS_Selection();

    template<typename T> void createContextInputBuffer( T* d_ptr, unsigned num_item, const char* name ); 
    optix::Group              createIAS(unsigned ias_idx);
    optix::Group              createIAS(const std::vector<qat4>& inst );
    optix::Group              createSolidSelectionIAS(unsigned ias_idx, const std::vector<unsigned>& solid_selection);

    optix::GeometryInstance   createGeometryInstance(unsigned solid_idx, unsigned identity);
    optix::Geometry           createGeometry(unsigned solid_idx);
    optix::Geometry           getGeometry(unsigned solid_idx) const ;  
 
    void setTop(const char* spec);
    void launch(unsigned width, unsigned height, unsigned depth);

    //void snap(const char* path, const char* bottom_line, const char* top_line, unsigned line_height) ;

};
