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
#include "PIP.h"
#include "SBT.h"

SBT::SBT(const PIP* pip_)
    :
    pip(pip_)
{
    init(); 
}

void SBT::init()
{
    std::cout << "SBT::init" << std::endl ; 
    createRaygen();

    createMiss();
    updateMiss(); 
}

/**
SBT::createMissSBT
--------------------

NB the records have opaque header and user data
**/

void SBT::createMiss()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss ), sizeof(Miss) ) );
    sbt.missRecordBase = d_miss;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip->miss_prog_group, &miss ) );

    sbt.missRecordStrideInBytes     = sizeof( Miss );
    sbt.missRecordCount             = 1;
}

void SBT::updateMiss()
{
    miss.data.r = 0.3f ;
    miss.data.g = 0.1f ;
    miss.data.b = 0.5f ;

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_miss ),
                &miss,
                sizeof(Miss),
                cudaMemcpyHostToDevice
                ) );
}

void SBT::createRaygen()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen ),   sizeof(Raygen) ) );
    sbt.raygenRecord = d_raygen;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip->raygen_prog_group,   &raygen ) );
}

void SBT::updateRaygen()
{
    std::cout <<  "SBT::updateRaygen tmin " << tmin << " tmax " << tmax << std::endl ; 

    raygen.data = {};

    raygen.data.cam_eye.x = eye.x ;
    raygen.data.cam_eye.y = eye.y ;
    raygen.data.cam_eye.z = eye.z ;

    raygen.data.camera_u.x = U.x ; 
    raygen.data.camera_u.y = U.y ; 
    raygen.data.camera_u.z = U.z ; 

    raygen.data.camera_v.x = V.x ; 
    raygen.data.camera_v.y = V.y ; 
    raygen.data.camera_v.z = V.z ; 

    raygen.data.camera_w.x = W.x ; 
    raygen.data.camera_w.y = W.y ; 
    raygen.data.camera_w.z = W.z ; 

    raygen.data.tmin = tmin ; 
    raygen.data.tmax = tmax ; 

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_raygen ),
                &raygen,
                sizeof( Raygen ),
                cudaMemcpyHostToDevice
                ) );
}

/**
TODO: move view params into constant memory referred from launch params rather than SBT data 
**/

void SBT::setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, float tmin_, float tmax_)
{
    std::cout <<  "PIP::setView tmin " << tmin << " tmax " << tmax << std::endl ; 
    eye = eye_ ; 
    U = U_ ; 
    V = V_ ; 
    W = W_ ; 
    tmin = tmin_ ; 
    tmax = tmax_ ; 

    updateRaygen();
}











void SBT::setGeo(const Geo* geo)
{
    createHitgroup(geo); 
}


void SBT::createHitgroup(const Geo* geo)
{
    unsigned num_gas = geo->getNumGAS(); 

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup ), sizeof(HitGroup)*num_gas ) );

    sbt.hitgroupRecordBase  = d_hitgroup;

    // allocate CPU side records 

    HitGroup* hg = new HitGroup[num_gas] ; 
    for(unsigned i=0 ; i < num_gas ; i++)
    { 
        OPTIX_CHECK( optixSbtRecordPackHeader( pip->hitgroup_prog_group, hg + i ) );
    }

    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroup );
    sbt.hitgroupRecordCount         = num_gas ;
    
    unsigned num_values = 1 ; 

    for(unsigned i=0 ; i < num_gas ; i++)
    {
        const GAS& gas = geo->getGAS(i) ;    

        // -------- dynamic SBT values ---------------------------   
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
        // --------------------------------------------------------

        (hg + i)->data.values = d_values ;  
        // sets device pointer into CPU struct about to be copied to device
        std::cout << "PIP::createHitgroup gas.extent " << gas.extent << std::endl ;  
    }

    // copy CPU side records to GPU 
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_hitgroup ),
                hg,
                sizeof( HitGroup )*num_gas,
                cudaMemcpyHostToDevice
                ) );

}



