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

#include "Geo.h"
#include "Binding.h"
#include "Params.h"

#include "PIP.h"
#include "SBT.h"

SBT::SBT(const PIP* pip_, Params* params_)
    :
    pip(pip_),
    params(params_),
    raygen(nullptr),
    miss(nullptr),
    hitgroup(nullptr)
{
    init(); 
}

void SBT::init()
{
    std::cout << "SBT::init" << std::endl ; 
    createRaygen();
    updateRaygen();

    createMiss();
    updateMiss(); 
}

void SBT::setGeo(const Geo* geo)
{
    createHitgroup(geo); 

    AS* top = geo->getTop();
    params->handle = top->handle ; 
}


/**
SBT::createMissSBT
--------------------

NB the records have opaque header and user data
**/

void SBT::createMiss()
{
    miss = new Miss ; 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss ), sizeof(Miss) ) );
    sbt.missRecordBase = d_miss;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip->miss_pg, miss ) );

    sbt.missRecordStrideInBytes     = sizeof( Miss );
    sbt.missRecordCount             = 1;
}

void SBT::updateMiss()
{
    miss->data.r = 0.3f ;
    miss->data.g = 0.1f ;
    miss->data.b = 0.5f ;

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_miss ),
                miss,
                sizeof(Miss),
                cudaMemcpyHostToDevice
                ) );
}

void SBT::createRaygen()
{
    raygen = new Raygen ; 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen ),   sizeof(Raygen) ) );
    sbt.raygenRecord = d_raygen;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip->raygen_pg,   raygen ) );
}

void SBT::updateRaygen()
{
    std::cout <<  "SBT::updateRaygen " << std::endl ; 

    raygen->data = {};

#ifdef SBT_VIEW
    raygen->data.eye.x = params->eye.x ;
    raygen->data.eye.y = params->eye.y ;
    raygen->data.eye.z = params->eye.z ;

    raygen->data.U.x = params->U.x ; 
    raygen->data.U.y = params->U.y ; 
    raygen->data.U.z = params->U.z ; 

    raygen->data.V.x = params->V.x ; 
    raygen->data.V.y = params->V.y ; 
    raygen->data.V.z = params->V.z ; 

    raygen->data.W.x = params->W.x ; 
    raygen->data.W.y = params->W.y ; 
    raygen->data.W.z = params->W.z ; 

    raygen->data.tmin = params->tmin ; 
    raygen->data.tmax = params->tmax ; 
#else
    raygen->data.placeholder = 42.0f ;
#endif

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_raygen ),
                raygen,
                sizeof( Raygen ),
                cudaMemcpyHostToDevice
                ) );
}


void SBT::createHitgroup(const Geo* geo)
{
    unsigned num_gas = geo->getNumGAS(); 
    hitgroup = new HitGroup[num_gas] ; 

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup ), sizeof(HitGroup)*num_gas ) );
    sbt.hitgroupRecordBase  = d_hitgroup;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroup);
    sbt.hitgroupRecordCount = num_gas ;

    for(unsigned i=0 ; i < num_gas ; i++)   // fill in the headers CPU side
         OPTIX_CHECK( optixSbtRecordPackHeader( pip->hitgroup_pg, hitgroup + i ) ); 
    
    unsigned num_values = 1 ; 

    for(unsigned i=0 ; i < num_gas ; i++)
    {
        const GAS& gas = geo->getGAS(i) ;    

        float* values = new float[num_values]; 
        for(unsigned i=0 ; i < num_values ; i++) values[i] = 0.f ; 
        values[0] = gas.extent ;  

        float* d_values = nullptr ; 
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_values ),
                    num_values*sizeof(float)
                    ) );

        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_values ),
                    values,
                    sizeof(float)*num_values,
                    cudaMemcpyHostToDevice
                    ) );

        delete [] values ; 

        (hitgroup + i)->data.values = d_values ; // set device pointer into CPU struct about to be copied to device
        std::cout << "PIP::createHitgroup gas.extent " << gas.extent << std::endl ;  
    }

    CUDA_CHECK( cudaMemcpy(      // copy CPU side records to GPU 
                reinterpret_cast<void*>( d_hitgroup ),
                hitgroup,
                sizeof( HitGroup )*num_gas,
                cudaMemcpyHostToDevice
                ) );
}

