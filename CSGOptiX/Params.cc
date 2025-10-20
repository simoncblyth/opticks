#include "SRG.h"
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

void Params::setPIDXYZ(unsigned x, unsigned y, unsigned z)
{
    pidxyz.x = x ;
    pidxyz.y = y ;
    pidxyz.z = z ;
}


/**
Params::setView
-----------------

Canonical invokation from CSGOptiX::prepareParamRender

**/

void Params::setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, const glm::vec3& WNORM_ )
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

    WNORM.x = WNORM_.x ;
    WNORM.y = WNORM_.y ;
    WNORM.z = WNORM_.z ;

}


void Params::setCamera(
    float tmin_,
    float tmax_,
    unsigned cameratype_,
    int traceyflip_,
    int rendertype_,
    const glm::vec4& ZPROJ_ )
{
    tmin = tmin_ ;
    tmax = tmax_ ;
    cameratype = cameratype_ ;
    traceyflip = traceyflip_ ;
    rendertype = rendertype_ ;

    ZPROJ.x = ZPROJ_.x ;
    ZPROJ.y = ZPROJ_.y ;
    ZPROJ.z = ZPROJ_.z ;
    ZPROJ.w = ZPROJ_.w ;
}

std::string Params::desc() const
{
    std::stringstream ss ;
    ss << "Params::desc"
       << std::endl
       << std::setw(20) << " raygenmode " << std::setw(10) << raygenmode  << std::endl
       << std::setw(20) << " handle " << std::setw(10) << handle  << std::endl
       << std::setw(20) << " width " << std::setw(10) << width  << std::endl
       << std::setw(20) << " height " << std::setw(10) << height  << std::endl
       << std::setw(20) << " depth " << std::setw(10) << depth  << std::endl
       << std::setw(20) << " cameratype " << std::setw(10) << cameratype  << std::endl
       << std::setw(20) << " traceyflip " << std::setw(10) << traceyflip  << std::endl
       << std::setw(20) << " rendertype " << std::setw(10) << rendertype  << std::endl
       << std::setw(20) << " origin_x " << std::setw(10) << origin_x  << std::endl
       << std::setw(20) << " origin_y " << std::setw(10) << origin_y  << std::endl
       << std::setw(20) << " tmin " << std::setw(10) << tmin  << std::endl
       << std::setw(20) << " tmax " << std::setw(10) << tmax  << std::endl
       ;

    std::string s = ss.str();
    return s ;
}

std::string Params::detail() const
{
    std::stringstream ss ;
    ss
        << "Params::detail"
        << std::endl
        << "(values)" << std::endl
        << desc()
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
    std::string s = ss.str();
    return s ;
}


Params::Params(int raygenmode_, unsigned width, unsigned height, unsigned depth)
    :
    raygenmode(SRG_RENDER),
    node(nullptr),
    plan(nullptr),
    tran(nullptr),
    itra(nullptr),
#if OPTIX_VERSION < 70000
    handle(nullptr),
#else
    handle(0),
#endif
    pixels(nullptr),
    isect(nullptr),
    fphoton(nullptr),
    width(0),
    height(0),
    depth(0),
    cameratype(0),
    traceyflip(0),
    origin_x(0),
    origin_y(0),
    tmin(0.f),
    tmin0(0.f),
    PropagateEpsilon0Mask(0u),
    tmax(0.f),
    vizmask(0xff),
    sim(nullptr),
    evt(nullptr),
    event_index(0),
    photon_slot_offset(0ull),
    max_time(1.e27f)
{
    setRaygenMode(raygenmode_);
    setSize(width, height, depth);
    setPIDXYZ(-1,-1,-1);     // disabled dumping (which WITH_PIDX enabled via setting to high unsigned:-1 values)
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

void Params::setVizmask(unsigned vizmask_)
{
    vizmask = vizmask_ ;
}

void Params::set_photon_slot_offset(unsigned long long photon_slot_offset_)
{
    photon_slot_offset = photon_slot_offset_ ;
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

