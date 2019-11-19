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

/**
UseOptiX7GeometryStandalone
==================================

**/

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <fstream>
#include <vector>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>


#include "sutil_vec_math.h"
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "UseOptiX7GeometryStandalone.h"

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;







// Composition::getEyeUVW and examples/UseGeometryShader:getMVP
void getEyeUVW(const glm::vec4& ce, const unsigned width, const unsigned height, glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W )
{
    glm::vec3 tr(ce.x, ce.y, ce.z);  // ce is center-extent of model
    glm::vec3 sc(ce.w);
    glm::vec3 isc(1.f/ce.w);
    // model frame unit coordinates from/to world 
    glm::mat4 model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    //glm::mat4 world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);

   // View::getTransforms
    glm::vec4 eye_m( -1.f,-1.f,1.f,1.f);  //  viewpoint in unit model frame 
    glm::vec4 look_m( 0.f, 0.f,0.f,1.f); 
    glm::vec4 up_m(   0.f, 0.f,1.f,1.f); 
    glm::vec4 gze_m( look_m - eye_m ) ; 

    const glm::mat4& m2w = model2world ; 
    glm::vec3 eye_ = glm::vec3( m2w * eye_m ) ; 
    //glm::vec3 look = glm::vec3( m2w * look_m ) ; 
    glm::vec3 up = glm::vec3( m2w * up_m ) ; 
    glm::vec3 gaze = glm::vec3( m2w * gze_m ) ;    

    glm::vec3 forward_ax = glm::normalize(gaze);
    glm::vec3 right_ax   = glm::normalize(glm::cross(forward_ax,up)); 
    glm::vec3 top_ax     = glm::normalize(glm::cross(right_ax,forward_ax));

    float aspect = float(width)/float(height) ;
    float tanYfov = 1.f ;  // reciprocal of camera zoom
    float gazelength = glm::length( gaze ) ;
    float v_half_height = gazelength * tanYfov ;
    float u_half_width  = v_half_height * aspect ;

    U = right_ax * u_half_width ;
    V = top_ax * v_half_height ;
    W = forward_ax * gazelength ; 
    eye = eye_ ; 
}


const char* PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext=".cu" )
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ptx/"
       << cmake_target
       << "_generated_"
       << cu_stem
       << cu_ext
       << ".ptx" 
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}


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




const char* PPMPath( const char* install_prefix, const char* stem, const char* ext=".ppm" )
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ppm/"
       << stem
       << ext
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}

void SPPM_write( const char* filename, const unsigned char* image, int width, int height, int ncomp, bool yflip )
{
    FILE * fp; 
    fp = fopen(filename, "wb");

    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned size = height*width*3 ; 
    unsigned char* data = new unsigned char[size] ; 

    for( int h=0; h < height ; h++ ) // flip vertically
    {   
        int y = yflip ? height - 1 - h : h ; 

        for( int x=0; x < width ; ++x ) 
        {
            *(data + (y*width+x)*3+0) = image[(h*width+x)*ncomp+0] ;   
            *(data + (y*width+x)*3+1) = image[(h*width+x)*ncomp+1] ;   
            *(data + (y*width+x)*3+2) = image[(h*width+x)*ncomp+2] ;   
        }
    }   
    fwrite(data, sizeof(unsigned char)*size, 1, fp);
    fclose(fp);  
    std::cout << "Wrote file (unsigned char*) " << filename << std::endl  ;
    delete[] data;
}



void SPPM_write( const char* filename, const uchar4* image, int width, int height, bool yflip )
{
    FILE * fp; 
    fp = fopen(filename, "wb");

    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned size = height*width*3 ; 
    unsigned char* data = new unsigned char[size] ; 

    for( int h=0; h < height ; h++ ) // flip vertically
    {   
        int y = yflip ? height - 1 - h : h ; 

        for( int x=0; x < width ; ++x ) 
        {
            *(data + (y*width+x)*3+0) = image[(h*width+x)].x ;   
            *(data + (y*width+x)*3+1) = image[(h*width+x)].y ;   
            *(data + (y*width+x)*3+2) = image[(h*width+x)].z ;   
        }
    }   
    fwrite(data, sizeof(unsigned char)*size, 1, fp);
    fclose(fp);  
    std::cout << "Wrote file (uchar4) " << filename << std::endl  ;
    delete[] data;
}















static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}


int main(int argc, char** argv)
{
    const char* name = "UseOptiX7GeometryStandalone" ; 
    const char* prefix = getenv("PREFIX"); 
    assert( prefix && "expecting PREFIX envvar pointing to writable directory" );

    const char* cmake_target = name ; 
    const char* ptx_path = PTXPath( prefix, cmake_target, name ) ; 

    std::cout << " ptx_path " << ptx_path << std::endl ; 

    unsigned width = 1024u ; 
    unsigned height = 768 ; 

    glm::vec4 ce(0.f,0.f,0.f, 3.f); 

    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 
    getEyeUVW( ce, width, height, eye, U, V, W ); 

    std::cout << argv[0] << std::endl ;  


    OptixDeviceContext context = nullptr;
    char log[2048]; // For error reporting from OptiX creation functions
    {
        CUDA_CHECK( cudaFree( 0 ) ); // Initialize CUDA

        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK( optixInit() );
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &context_log_cb;
        options.logCallbackLevel          = 4;
        OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
    }


    // accel handling
    //
    OptixTraversableHandle gas_handle;
    CUdeviceptr            d_gas_output_buffer;
    {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

        // AABB build input
        OptixAabb   aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
        CUdeviceptr d_aabb_buffer;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_aabb_buffer ),
                    &aabb,
                    sizeof( OptixAabb ),
                    cudaMemcpyHostToDevice
                    ) );

        OptixBuildInput aabb_input = {};

        aabb_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        aabb_input.aabbArray.aabbBuffers   = &d_aabb_buffer;
        aabb_input.aabbArray.numPrimitives = 1;

        uint32_t aabb_input_flags[1]       = {OPTIX_GEOMETRY_FLAG_NONE};
        aabb_input.aabbArray.flags         = aabb_input_flags;
        aabb_input.aabbArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &aabb_input, 1, &gas_buffer_sizes ) );
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                    compactedSizeOffset + 8
                    ) );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

        OPTIX_CHECK( optixAccelBuild( context,
                                      0,                  // CUDA stream
                                      &accel_options,
                                      &aabb_input,
                                      1,                  // num build inputs
                                      d_temp_buffer_gas,
                                      gas_buffer_sizes.tempSizeInBytes,
                                      d_buffer_temp_output_gas_and_compacted_size,
                                      gas_buffer_sizes.outputSizeInBytes,
                                      &gas_handle,
                                      &emitProperty,      // emitted property list
                                      1                   // num emitted properties
                                      ) );

        CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
        CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );

        size_t compacted_gas_size;
        CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

            CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
        }
        else
        {
            d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }


    //
    // Create module
    //

    std::string ptx ; 
    readFile(ptx, ptx_path ); 

    std::cout 
        << " ptx_path " << ptx_path << std::endl 
        << " ptx size " << ptx.size() << std::endl 
        ;

    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues      = 3;
        pipeline_compile_options.numAttributeValues    = 3;
        pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        size_t sizeof_log = sizeof( log );

        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                    context,
                    &module_compile_options,
                    &pipeline_compile_options,
                    ptx.c_str(),
                    ptx.size(),
                    log,
                    &sizeof_log,
                    &module
                    ) );
    }


    //
    // Create program groups
    //

    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    {
        OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
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
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
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
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
                    &hitgroup_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &hitgroup_prog_group
                    ) );
    }


    //
    // Link pipeline
    //

    OptixPipeline pipeline = nullptr;
    {
        OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth          = 5;
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        pipeline_link_options.overrideUsesMotionBlur = false;
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixPipelineCreate(
                    context,
                    &pipeline_compile_options,
                    &pipeline_link_options,
                    program_groups,
                    sizeof( program_groups ) / sizeof( program_groups[0] ),
                    log,
                    &sizeof_log,
                    &pipeline
                    ) );
    }


    //
    // Set up shader binding table
    //

    OptixShaderBindingTable sbt = {};
    {
        CUdeviceptr  raygen_record;
        const size_t raygen_record_size = sizeof( RayGenSbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );


        RayGenSbtRecord rg_sbt;
        rg_sbt.data ={};

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


        OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( raygen_record ),
                    &rg_sbt,
                    raygen_record_size,
                    cudaMemcpyHostToDevice
                    ) );

        CUdeviceptr miss_record;
        size_t      miss_record_size = sizeof( MissSbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
        MissSbtRecord ms_sbt;
        ms_sbt.data = { 0.3f, 0.1f, 0.2f };
        OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( miss_record ),
                    &ms_sbt,
                    miss_record_size,
                    cudaMemcpyHostToDevice
                    ) );

        CUdeviceptr hitgroup_record;
        size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
        HitGroupSbtRecord hg_sbt;
        hg_sbt.data = { 1.5f };
        OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( hitgroup_record ),
                    &hg_sbt,
                    hitgroup_record_size,
                    cudaMemcpyHostToDevice
                    ) );

        sbt.raygenRecord                = raygen_record;
        sbt.missRecordBase              = miss_record;
        sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
        sbt.missRecordCount             = 1;
        sbt.hitgroupRecordBase          = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
        sbt.hitgroupRecordCount         = 1;
    }


    // alloc output buffer

    std::vector<uchar4> host_pixels ; 
    uchar4* device_pixels ; 

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( device_pixels ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &device_pixels ),
                width*height*sizeof(uchar4)
                ) );


    //
    // launch
    //
    {
        CUstream stream;
        CUDA_CHECK( cudaStreamCreate( &stream ) );

        Params params;
        params.image        = device_pixels ;
        params.image_width  = width;
        params.image_height = height;
        params.origin_x     = width / 2;
        params.origin_y     = height / 2;
        params.handle       = gas_handle;

        CUdeviceptr d_param;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_param ),
                    &params, sizeof( params ),
                    cudaMemcpyHostToDevice
                    ) );

        OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
        CUDA_SYNC_CHECK();
    }



    host_pixels.resize(width*height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( host_pixels.data() ),
                device_pixels,
                width*height*sizeof(uchar4),
                cudaMemcpyDeviceToHost
    ));


    const char* ppm_path = PPMPath( prefix, name ); 
    std::cout << "write ppm_path " << ppm_path << std::endl ; 
 
    bool yflip = true ;  
    SPPM_write( ppm_path,  host_pixels.data() , width, height, yflip );

    return 0 ; 
}

