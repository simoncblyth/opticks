#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <fstream>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

#include "Engine.h"
#include "Geo.h"
#include "Binding.h"
#include "PIP.h"


static bool readFile( std::string& str, const char* path )
{
    std::ifstream fp(path);
    if( fp.good() )
    {   
        std::stringstream content ;
        content << fp.rdbuf();
        str = content.str();
        return true;
    }   
    return false;
}

OptixPipelineCompileOptions PIP::CreateOptions(unsigned numPayloadValues, unsigned numAttributeValues ) // static
{
    OptixPipelineCompileOptions pipeline_compile_options = {} ;

    pipeline_compile_options.usesMotionBlur        = false;
    //pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING ; 
    // without the OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING  got no intersects

    pipeline_compile_options.numPayloadValues      = numPayloadValues ;   // in optixTrace call
    pipeline_compile_options.numAttributeValues    = numAttributeValues ;
    //pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE; 
    pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    return pipeline_compile_options ;  
}

 
PIP::PIP(const char* ptx_path_) 
    :
    geo(Geo::Get()),
    num_gas(geo->getNumGAS()), 
    pipeline_compile_options(CreateOptions(4,3)),
    module(CreateModule(ptx_path_,pipeline_compile_options))
{
    init(); 
}


void PIP::init()
{
    std::cout << "PIP::init "
              << " num_gas " << num_gas
              << std::endl 
              ;

    createProgramGroups();
    linkPipeline();
    createSbt(); 
}

/**
PIP::createModule
-------------------

PTX from file is read and compiled into the module

**/

OptixModule PIP::CreateModule(const char* ptx_path, OptixPipelineCompileOptions& pipeline_compile_options ) // static 
{
    std::string ptx ; 
    readFile(ptx, ptx_path ); 

    std::cout 
        << " ptx_path " << ptx_path << std::endl 
        << " ptx size " << ptx.size() << std::endl 
        ;

    OptixModule module = nullptr ;

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    size_t sizeof_log = 0 ; 
    char log[2048]; // For error reporting from OptiX creation functions

    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                Engine::context,
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
                ) );

    return module ; 
}


/**
PIP::createProgramGroups
--------------------------

Associates function names from module(s) into OptixProgramGroup

Thoughts
~~~~~~~~~~~

Twas initially thinking of needing multiple
intersection programs for different shapes, but its 
closer to OptiX6 to just have one equivalent to 
oxrap/cu/intersect_analytic.cu:intersect 

which acts as the shape ordered by the contents of the SBT 


**/

void PIP::createProgramGroups()
{
    size_t sizeof_log = 0 ; 
    char log[2048]; // For error reporting from OptiX creation functions

    OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                Engine::context,
                &raygen_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &raygen_prog_group
                ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                Engine::context,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &miss_prog_group
                ) );

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
    hitgroup_prog_group_desc.hitgroup.moduleIS            = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                Engine::context,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_prog_group
                ) );
}

/**
PIP::linkPipeline
-------------------

Create pipeline from the program_groups

**/

void PIP::linkPipeline()
{
    size_t sizeof_log = 0 ; 
    char log[2048]; // For error reporting from OptiX creation functions

    OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = 5;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipeline_link_options.overrideUsesMotionBlur = false;

    OPTIX_CHECK_LOG( optixPipelineCreate(
                Engine::context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                log,
                &sizeof_log,
                &pipeline
                ) );
}

/**
PIP::createSbt
-------------------------------

p31 
    The program groups are also used to initialize the header of the SBT record
    associated with those programs.

**/


void PIP::createRayGenSbt()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ),   sizeof(RayGenSbtRecord) ) );
    sbt.raygenRecord        = raygen_record;
    OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group,   &rg_sbt ) );
    // presumably the pipeline always has 1 RayGenRecord so no need for counts ?
}

void PIP::createMissSbt()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ),     sizeof(MissSbtRecord) ) );
    sbt.missRecordBase      = miss_record;
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group,     &ms_sbt ) );


    sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    sbt.missRecordCount             = 1;
}

void PIP::createHitGroupSbt_0()
{
    assert(0);  // this only works for a single gas
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), sizeof(HitGroupSbtRecord)*num_gas ) );

    sbt.hitgroupRecordBase  = hitgroup_record;

    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
    assert( num_gas == 1 ); 

    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    sbt.hitgroupRecordCount         = num_gas ;

    assert( num_gas == 1 ); 
    float extent = geo->getGAS_Extent(0) ;
    hg_sbt.data.radius = extent ;

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( hitgroup_record ),
                &hg_sbt,
                sizeof( HitGroupSbtRecord )*num_gas,
                cudaMemcpyHostToDevice
                ) );

}

void PIP::createHitGroupSbt()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), sizeof(HitGroupSbtRecord)*num_gas ) );

    sbt.hitgroupRecordBase  = hitgroup_record;

    HitGroupSbtRecord* hg_sbt_ptr = new HitGroupSbtRecord[num_gas] ; 
    for(unsigned i=0 ; i < num_gas ; i++)
    { 
        OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, hg_sbt_ptr + i ) );
    }

    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    sbt.hitgroupRecordCount         = num_gas ;

    for(unsigned i=0 ; i < num_gas ; i++)
    {
        float extent = geo->getGAS_Extent(i) ;  
        (hg_sbt_ptr + i)->data.radius = extent ;
        std::cout << "PIP::createHitGroupSbt extent " << extent << std::endl ;  
    }

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( hitgroup_record ),
                hg_sbt_ptr,
                sizeof( HitGroupSbtRecord )*num_gas,
                cudaMemcpyHostToDevice
                ) );

}

void PIP::createSbt()
{
    std::cout << "PIP::createSbt num_gas " << num_gas << std::endl ; 
    createRayGenSbt();
    createMissSbt();
    createHitGroupSbt(); 
}

 
void PIP::updateHitGroupSbt()
{
    std::cout << "PIP::updateHitGroupSbt num_gas " << num_gas << " nothing to do : geo is constant " << std::endl ; 
}

void PIP::updateMissSbt()
{
    ms_sbt.data = { 0.3f, 0.1f, 0.2f };

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( miss_record ),
                &ms_sbt,
                sizeof(MissSbtRecord),
                cudaMemcpyHostToDevice
                ) );
}


/**
PIP::updateSbt
------------------------------

1. populate the CPU side sbt records
2. upload sbt records 

**/

void PIP::updateSbt()
{
    updateRayGenSbt();
    updateMissSbt();
    updateHitGroupSbt();
}


void PIP::updateRayGenSbt()
{
    rg_sbt.data = {};

    rg_sbt.data.cam_eye.x = eye.x ;
    rg_sbt.data.cam_eye.y = eye.y ;
    rg_sbt.data.cam_eye.z = eye.z ;

    rg_sbt.data.camera_u.x = U.x ; 
    rg_sbt.data.camera_u.y = U.y ; 
    rg_sbt.data.camera_u.z = U.z ; 

    rg_sbt.data.camera_v.x = V.x ; 
    rg_sbt.data.camera_v.y = V.y ; 
    rg_sbt.data.camera_v.z = V.z ; 

    rg_sbt.data.camera_w.x = W.x ; 
    rg_sbt.data.camera_w.y = W.y ; 
    rg_sbt.data.camera_w.z = W.z ; 

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( raygen_record ),
                &rg_sbt,
                sizeof( RayGenSbtRecord ),
                cudaMemcpyHostToDevice
                ) );
}


void PIP::setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_)
{
    eye = eye_ ; 
    U = U_ ; 
    V = V_ ; 
    W = W_ ; 

    updateSbt(); 
}

