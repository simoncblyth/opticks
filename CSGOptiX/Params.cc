#include "SRG.h"
#include "Params.h"

#ifndef __CUDACC__

#include <iostream>
#include <iomanip>
#include "CUDA_CHECK.h"
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <iostream>


Params_Helper::Params_Helper( Params* _params, int raygenmode_, unsigned width, unsigned height, unsigned depth )
   :
   params(_params),
   d_params(nullptr),
   level(0)
{
   init();

   setRaygenMode(raygenmode_);
   setSize(width, height, depth);
   setPIDXYZ(-1,-1,-1,0);     // disabled dumping (which WITH_PIDX enabled via setting to high unsigned:-1 values)
}

void Params_Helper::init()
{
    params->node = nullptr ;
    params->plan = nullptr ;
    params->tran = nullptr ;
    params->itra = nullptr ;
    params->pixels = nullptr ;
    params->isect = nullptr ;
    params->fphoton = nullptr ;
    params->sim = nullptr ;
    params->evt = nullptr ;
#if OPTIX_VERSION < 70000
    params->handle = nullptr ;
#else
    params->handle = 0 ;
#endif
    params->photon_slot_offset = 0ull ;
    params->raygenmode = SRG_RENDER ;
    params->width = 0 ;
    params->height = 0 ;
    params->depth = 0 ;
    params->cameratype = 0 ;
    params->traceyflip = 0 ;
    params->origin_x = 0 ;
    params->origin_y = 0 ;
    params->event_index = 0 ;
    params->tmin = 0.f;
    params->tmin0 = 0.f;
    params->PropagateRefineDistance = 0.f;
    params->tmax = 0.f ;
    params->max_time = 1.e27f;
    params->PropagateEpsilon0Mask = 0u;
    params->vizmask = 0xff ;
    params->PropagateRefine = 0u ;
}


void Params_Helper::setCenterExtent(float x, float y, float z, float w)  // used for "simulation" planar rendering
{
    params->center_extent.x = x ;
    params->center_extent.y = y ;
    params->center_extent.z = z ;
    params->center_extent.w = w ;
}

void Params_Helper::setPIDXYZ(unsigned x, unsigned y, unsigned z, unsigned w)
{
    params->pidxyz.x = x ;
    params->pidxyz.y = y ;
    params->pidxyz.z = z ;
    params->pidxyz.w = w ;
}


/**
Params_Helper::setView
-----------------

Canonical invokation from CSGOptiX::prepareParamRender

**/

void Params_Helper::setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, const glm::vec3& WNORM_ )
{
    params->eye.x = eye_.x ;
    params->eye.y = eye_.y ;
    params->eye.z = eye_.z ;
    params->eye.w = 0.f ;

    params->U.x = U_.x ;
    params->U.y = U_.y ;
    params->U.z = U_.z ;
    params->U.w = 0.f ;

    params->V.x = V_.x ;
    params->V.y = V_.y ;
    params->V.z = V_.z ;
    params->V.w = 0.f ;

    params->W.x = W_.x ;
    params->W.y = W_.y ;
    params->W.z = W_.z ;
    params->W.w = 0.f ;

    params->WNORM.x = WNORM_.x ;
    params->WNORM.y = WNORM_.y ;
    params->WNORM.z = WNORM_.z ;
    params->WNORM.w = 0.f ;

}


void Params_Helper::setCamera(
    float tmin_,
    float tmax_,
    unsigned cameratype_,
    int traceyflip_,
    int rendertype_,
    const glm::vec4& ZPROJ_ )
{
    params->tmin = tmin_ ;
    params->tmax = tmax_ ;
    params->cameratype = cameratype_ ;
    params->traceyflip = traceyflip_ ;
    params->rendertype = rendertype_ ;

    params->ZPROJ.x = ZPROJ_.x ;
    params->ZPROJ.y = ZPROJ_.y ;
    params->ZPROJ.z = ZPROJ_.z ;
    params->ZPROJ.w = ZPROJ_.w ;
}

std::string Params_Helper::desc() const
{
    std::stringstream ss ;
    ss << "Params_Helper::desc"
       << std::endl
       << std::setw(20) << " raygenmode " << std::setw(10) << params->raygenmode  << std::endl
       << std::setw(20) << " handle " << std::setw(10) << params->handle  << std::endl
       << std::setw(20) << " width " << std::setw(10) << params->width  << std::endl
       << std::setw(20) << " height " << std::setw(10) << params->height  << std::endl
       << std::setw(20) << " depth " << std::setw(10) << params->depth  << std::endl
       << std::setw(20) << " cameratype " << std::setw(10) << params->cameratype  << std::endl
       << std::setw(20) << " traceyflip " << std::setw(10) << params->traceyflip  << std::endl
       << std::setw(20) << " rendertype " << std::setw(10) << params->rendertype  << std::endl
       << std::setw(20) << " origin_x " << std::setw(10) << params->origin_x  << std::endl
       << std::setw(20) << " origin_y " << std::setw(10) << params->origin_y  << std::endl
       << std::setw(20) << " tmin " << std::setw(10) << params->tmin  << std::endl
       << std::setw(20) << " tmax " << std::setw(10) << params->tmax  << std::endl
       << std::setw(20) << " eye("
       << std::setw(10) << std::fixed << std::setprecision(6) << params->eye.x
       << std::setw(10) << std::fixed << std::setprecision(6) << params->eye.y
       << std::setw(10) << std::fixed << std::setprecision(6) << params->eye.z
       << std::setw(10) << std::fixed << std::setprecision(6) << params->eye.w
       << ")\n"
       << std::setw(20) << " U("
       << std::setw(10) << std::fixed << std::setprecision(6) << params->U.x
       << std::setw(10) << std::fixed << std::setprecision(6) << params->U.y
       << std::setw(10) << std::fixed << std::setprecision(6) << params->U.z
       << std::setw(10) << std::fixed << std::setprecision(6) << params->U.w
       << ")\n"
       << std::setw(20) << " V("
       << std::setw(10) << std::fixed << std::setprecision(6) << params->V.x
       << std::setw(10) << std::fixed << std::setprecision(6) << params->V.y
       << std::setw(10) << std::fixed << std::setprecision(6) << params->V.z
       << std::setw(10) << std::fixed << std::setprecision(6) << params->V.w
       << ")\n"
       << std::setw(20) << " W("
       << std::setw(10) << std::fixed << std::setprecision(6) << params->W.x
       << std::setw(10) << std::fixed << std::setprecision(6) << params->W.y
       << std::setw(10) << std::fixed << std::setprecision(6) << params->W.z
       << std::setw(10) << std::fixed << std::setprecision(6) << params->W.w
       << ")\n"
       << std::setw(20) << " WNORM("
       << std::setw(10) << std::fixed << std::setprecision(6) << params->WNORM.x
       << std::setw(10) << std::fixed << std::setprecision(6) << params->WNORM.y
       << std::setw(10) << std::fixed << std::setprecision(6) << params->WNORM.z
       << std::setw(10) << std::fixed << std::setprecision(6) << params->WNORM.w
       << ")\n"
       ;

    std::string str = ss.str();
    return str ;
}

std::string Params_Helper::detail() const
{
    std::stringstream ss ;
    ss
        << "Params_Helper::detail"
        << std::endl
        << "(values)" << std::endl
        << desc()
        << std::endl
        << "(device pointers)" << std::endl
        << std::setw(20) << " node " << std::setw(10) << params->node  << std::endl
        << std::setw(20) << " plan " << std::setw(10) << params->plan  << std::endl
        << std::setw(20) << " tran " << std::setw(10) << params->tran  << std::endl
        << std::setw(20) << " itra " << std::setw(10) << params->itra  << std::endl
        << std::setw(20) << " pixels " << std::setw(10) << params->pixels  << std::endl
        << std::setw(20) << " isect " << std::setw(10) << params->isect  << std::endl
        << std::setw(20) << " sim " << std::setw(10) << params->sim  << std::endl
        << std::setw(20) << " evt " << std::setw(10) << params->evt  << std::endl
        ;
    std::string str = ss.str();
    return str ;
}



void Params_Helper::setRaygenMode(int raygenmode_)
{
    params->raygenmode = raygenmode_ ;
}

void Params_Helper::setSize(unsigned width_, unsigned height_, unsigned depth_ )
{
    params->width = width_ ;
    params->height = height_ ;
    params->depth = depth_ ;

    params->origin_x = width_ / 2;
    params->origin_y = height_ / 2;
}

void Params_Helper::setVizmask(unsigned vizmask_)
{
    params->vizmask = vizmask_ ;
}

void Params_Helper::set_photon_slot_offset(unsigned long long photon_slot_offset_)
{
    params->photon_slot_offset = photon_slot_offset_ ;
}



void Params_Helper::device_alloc()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( Params ) ) );
    if(level > 0) std::cout << "Params_Helper::device_alloc d_params address is " << (void*)d_params << std::endl;
    assert( d_params );
}
void Params_Helper::upload()
{
    size_t sizeof_params = sizeof( *params );
    size_t sizeof_Params = sizeof( Params );
    bool consistent = sizeof_params == sizeof_Params ;

    if(level > 0) std::cout
        << "Params_Helper::upload d_params address is " << (void*)d_params
        << " sizeof_Params " << sizeof_Params
        << " sizeof_params " << sizeof_params
        << " consistent " << ( consistent ? "YES" : "NO " )
        << std::endl;

    assert( consistent );

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_params ), params, sizeof_params , cudaMemcpyHostToDevice) );
}

#endif

