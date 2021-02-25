
#include <iostream>
#include <iomanip>

#include "Ctx.h"
#include "AS.h"
#include "Params.h"

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

OptixDeviceContext Ctx::context = nullptr ;

void Ctx::context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)  // static 
{
    std::cerr 
        << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
        << message << "\n";
}

Ctx::Ctx()
    :
    params(new Params)
{
    CUDA_CHECK( cudaFree( 0 ) ); 

    CUcontext cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &Ctx::context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
}

void Ctx::setTop(const AS* top)
{
    params->handle = top->handle ; 
}


void Ctx::uploadParams()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                params, sizeof( Params ),
                cudaMemcpyHostToDevice
                ) );
}

void Ctx::setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, float tmin_, float tmax_, unsigned cameratype )
{
    params->eye.x = eye_.x ;
    params->eye.y = eye_.y ;
    params->eye.z = eye_.z ;

    params->U.x = U_.x ; 
    params->U.y = U_.y ; 
    params->U.z = U_.z ; 

    params->V.x = V_.x ; 
    params->V.y = V_.y ; 
    params->V.z = V_.z ; 

    params->W.x = W_.x ; 
    params->W.y = W_.y ; 
    params->W.z = W_.z ; 

    params->tmin = tmin_ ; 
    params->tmax = tmax_ ; 
    params->cameratype = cameratype ; 

    std::cout << "Ctx::setView"
              << " tmin " << tmin_  
              << " tmax " << tmax_
              << " cameratype " << cameratype
              << std::endl 
              ;  

}

void Ctx::setSize(unsigned width_, unsigned height_, unsigned depth_ )
{
    params->width = width_ ;
    params->height = height_ ;
    params->depth = depth_ ;

    params->origin_x = width_ / 2;
    params->origin_y = height_ / 2;
}


