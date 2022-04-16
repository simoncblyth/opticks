#pragma once
/**
Six : backward compatibility layer :  enabling new workflow CSGFoundry geometry to be used with OptiX < 7 
===========================================================================================================

**/

#include <map>
#include <vector>
#include <optixu/optixpp_namespace.h>

struct qat4 ; 

class Opticks ; 

struct CSGFoundry ; 
struct CSGSolid ; 

struct Params ; 

struct Six
{
    const Opticks*          ok ; 
    const std::vector<unsigned>&  solid_selection ; 
    unsigned long long      emm ; 

    optix::Context     context ;
    optix::Material    material ;

    // TODO: see if these can be replaced by CUDA buffers from Params ?
    optix::Buffer      pixels_buffer ; 
    optix::Buffer      posi_buffer ; 
    optix::Buffer      isect_buffer ; 

    Params*           params ; 
    const char*       ptx_path ; 
    const char*       geo_ptx_path ; 
    unsigned          entry_point_index ; 
    unsigned          optix_device_ordinal ; 
    const CSGFoundry* foundry ; 
    int               pindex ; 

    std::map<unsigned, optix::Geometry> solids ; 

    std::vector<optix::Group>    groups ; 

    Six(const Opticks* ok, const char* ptx_path, const char* geo_ptx_path, Params* params_);  
    void initContext();
    void initFrame();   // hookup pixels and isect buffers

    void updateContext();  // for changed params such as viewpoint 
    void initPipeline();
    void setFoundry(const CSGFoundry* foundry);

    void createGeom();
    void createContextBuffers();
    void createGAS();
    void createGAS_Standard();
    void createGAS_Selection();
    void createIAS();
    void createIAS_Standard();
    void createIAS_Selection();

    template<typename T> void createContextBuffer( char typ, T* d_ptr, unsigned num_item, const char* name ); 
    optix::Group              createIAS(unsigned ias_idx);
    optix::Group              createIAS(const std::vector<qat4>& inst );
    optix::Group              createSolidSelectionIAS(unsigned ias_idx, const std::vector<unsigned>& solid_selection);

    optix::GeometryInstance   createGeometryInstance(unsigned solid_idx, unsigned identity);
    optix::Geometry           createGeometry(unsigned solid_idx);
    optix::Geometry           getGeometry(unsigned solid_idx) const ;  
 
    void setTop(const char* spec);
    void launch();

    void snap(const char* path, const char* bottom_line, const char* top_line, unsigned line_height) ;

};
