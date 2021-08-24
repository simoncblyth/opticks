#include "Params.h"

#ifndef __CUDACC__

#include <iostream>
#include <iomanip>
#include "CUDA_CHECK.h"
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <iostream>


void Params::setCenterExtent(float x, float y, float z, float w)  // used for "simulation" planar rendering 
{
    center_extent.x = x ; 
    center_extent.y = y ; 
    center_extent.z = z ; 
    center_extent.w = w ; 
}

void Params::setCEGS(const uint4& _cegs)
{
    cegs.x = _cegs.x ; 
    cegs.y = _cegs.y ; 
    cegs.z = _cegs.z ; 
    cegs.w = _cegs.w ; 
}

void Params::setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_ )
{
    eye.x = eye_.x ;
    eye.y = eye_.y ;
    eye.z = eye_.z ;

    U.x = U_.x ; 
    U.y = U_.y ; 
    U.z = U_.z ; 

    V.x = V_.x ; 
    V.y = V_.y ; 
    V.z = V_.z ; 

    W.x = W_.x ; 
    W.y = W_.y ; 
    W.z = W_.z ; 
}

void Params::setView(const glm::vec4& eye_, const glm::vec4& U_, const glm::vec4& V_, const glm::vec4& W_)
{
    eye.x = eye_.x ;
    eye.y = eye_.y ;
    eye.z = eye_.z ;

    U.x = U_.x ; 
    U.y = U_.y ; 
    U.z = U_.z ; 

    V.x = V_.x ; 
    V.y = V_.y ; 
    V.z = V_.z ; 

    W.x = W_.x ; 
    W.y = W_.y ; 
    W.z = W_.z ; 
}

void Params::setCamera(float tmin_, float tmax_, unsigned cameratype_ )
{
    tmin = tmin_ ; 
    tmax = tmax_ ; 
    cameratype = cameratype_ ; 

    std::cout << "Params::setCamera"
              << " tmin " << tmin  
              << " tmax " << tmax
              << " cameratype " << cameratype
              << std::endl 
              ;  

}

void Params::dump(const char* msg) const 
{
    std::cout 
        << msg << std::endl 
        << std::endl 
        << "(values)" << std::endl 
        << std::setw(20) << " raygenmode " << std::setw(10) << raygenmode  << std::endl 
        << std::setw(20) << " handle " << std::setw(10) << handle  << std::endl 
        << std::setw(20) << " width " << std::setw(10) << width  << std::endl 
        << std::setw(20) << " height " << std::setw(10) << height  << std::endl 
        << std::setw(20) << " depth " << std::setw(10) << depth  << std::endl 
        << std::setw(20) << " cameratype " << std::setw(10) << cameratype  << std::endl 
        << std::setw(20) << " origin_x " << std::setw(10) << origin_x  << std::endl 
        << std::setw(20) << " origin_y " << std::setw(10) << origin_y  << std::endl 
        << std::setw(20) << " tmin " << std::setw(10) << tmin  << std::endl 
        << std::setw(20) << " tmax " << std::setw(10) << tmax  << std::endl 
        << std::setw(20) << " num_photons " << std::setw(10) << num_photons  << std::endl 
        << std::endl 
        << "(device pointers)" << std::endl 
        << std::setw(20) << " node " << std::setw(10) << node  << std::endl 
        << std::setw(20) << " plan " << std::setw(10) << plan  << std::endl 
        << std::setw(20) << " tran " << std::setw(10) << tran  << std::endl 
        << std::setw(20) << " itra " << std::setw(10) << itra  << std::endl 
        << std::setw(20) << " pixels " << std::setw(10) << pixels  << std::endl 
        << std::setw(20) << " isect " << std::setw(10) << isect  << std::endl 
        << std::setw(20) << " sim " << std::setw(10) << sim  << std::endl 
        << std::setw(20) << " evt " << std::setw(10) << evt  << std::endl 
        ;
}


Params::Params(int raygenmode_, unsigned width, unsigned height, unsigned depth)
{
    setRaygenMode(raygenmode_); 
    setSize(width, height, depth); 
}

void Params::setRaygenMode(int raygenmode_)
{
    raygenmode = raygenmode_ ; 
}

void Params::setSize(unsigned width_, unsigned height_, unsigned depth_ )
{
    width = width_ ;
    height = height_ ;
    depth = depth_ ;

    origin_x = width_ / 2;
    origin_y = height_ / 2;
}


Params* Params::d_param = nullptr ; 

void Params::device_alloc()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );   
    assert( d_param ); 
}
void Params::upload()
{
    assert( d_param ); 
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_param ), this, sizeof( Params ), cudaMemcpyHostToDevice) ); 
}


#endif


