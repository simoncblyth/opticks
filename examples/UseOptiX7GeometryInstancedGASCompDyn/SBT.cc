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

/**
SBT
====

SBT needs PIP as the packing of SBT record headers requires 
access to their corresponding program groups (PGs).  
This is one aspect of establishing the connection between the 
PGs and their data.

**/

SBT::SBT(const PIP* pip_)
    :
    pip(pip_),
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
    checkHitgroup(); 
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
    raygen->data.placeholder = 42.0f ;

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
    unsigned tot_bi = 0 ; 
    for(unsigned i=0 ; i < num_gas ; i++)
    {
        const GAS& gas = geo->getGAS(i) ;    
        tot_bi += gas.bis.size() ;  
    }
    assert( tot_bi > 0 );  
    std::cout << "SBT::createHitgroup num_gas " << num_gas << " tot_bi " << tot_bi << std::endl ; 

    hitgroup = new HitGroup[tot_bi] ; 
    HitGroup* hg = hitgroup ; 
    unsigned  tot_bi2 = 0 ; 

    for(unsigned i=0 ; i < tot_bi ; i++)   // pack headers CPU side
         OPTIX_CHECK( optixSbtRecordPackHeader( pip->hitgroup_pg, hitgroup + i ) ); 
    
    for(unsigned i=0 ; i < num_gas ; i++)
    {
        const GAS& gas = geo->getGAS(i) ;    
        unsigned num_bi = gas.bis.size(); 
  
        std::cout << "SBT::createHitgroup gas_idx " << i << " num_bi " << num_bi << std::endl ; 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const float* aabb = bi.aabb ; 

            // how to organize generalization by primitive OR CSG tree type ?
            float radius = *(aabb+4) ;  
            unsigned num_items = 1 ; 
            float* values = new float[num_items]; 
            values[0] = radius  ;  
            float* d_values = UploadArray<float>(values, num_items) ; 
            delete [] values ; 
     
            hg->data.values = d_values ; // set device pointer into CPU struct about to be copied to device
            hg->data.bindex = j ;  

            hg++ ; 
            tot_bi2++ ; 
        }
    }
    assert( tot_bi == tot_bi2 );  

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup ), sizeof(HitGroup)*tot_bi ) );
    sbt.hitgroupRecordBase  = d_hitgroup;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroup);
    sbt.hitgroupRecordCount = tot_bi ;

    CUDA_CHECK( cudaMemcpy(   
                reinterpret_cast<void*>( d_hitgroup ),
                hitgroup,
                sizeof( HitGroup )*tot_bi,
                cudaMemcpyHostToDevice
                ) );
}


void SBT::checkHitgroup()
{
    std::cout 
        << "SBT::checkHitgroup" 
        << " sbt.hitgroupRecordCount " << sbt.hitgroupRecordCount
        << std::endl 
        ; 

    check = new HitGroup[sbt.hitgroupRecordCount] ; 

    CUDA_CHECK( cudaMemcpy(   
                check,
                reinterpret_cast<void*>( sbt.hitgroupRecordBase ),
                sizeof( HitGroup )*sbt.hitgroupRecordCount,
                cudaMemcpyDeviceToHost
                ) );

    for(unsigned i=0 ; i < sbt.hitgroupRecordCount ; i++)
    {
        HitGroup* hg = check + i  ; 
        unsigned num_items = 1 ; 
        float* d_values = hg->data.values ; 
        float* values = DownloadArray<float>(d_values, num_items);  

        std::cout << "SBT::checkHitgroup downloaded array, num_items " << num_items << " : " ; 
        for(unsigned j=0 ; j < num_items ; j++) std::cout << *(values+j) << " " ; 
        std::cout << std::endl ; 
    }
}


template <typename T>
T* SBT::UploadArray(const T* array, unsigned num_items ) // static
{
    T* d_array = nullptr ; 
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_array ),
                num_items*sizeof(T)
                ) );


    std::cout << "SBT::UploadArray num_items " << num_items << " : " ; 
    for(unsigned i=0 ; i < num_items ; i++) std::cout << *(array+i) << " " ; 
    std::cout << std::endl ; 

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_array ),
                array,
                sizeof(T)*num_items,
                cudaMemcpyHostToDevice
                ) );

    return d_array ; 
}


template <typename T>
T* SBT::DownloadArray(const T* d_array, unsigned num_items ) // static
{
    std::cout << "SBT::DownloadArray num_items " << num_items << " : " ; 

    T* array = new T[num_items] ;  
    CUDA_CHECK( cudaMemcpy(
                array,
                d_array,
                sizeof(T)*num_items,
                cudaMemcpyDeviceToHost
                ) );

    return array ; 
}




template float* SBT::UploadArray<float>(const float* array, unsigned num_items) ;
template float* SBT::DownloadArray<float>(const float* d_array, unsigned num_items) ;



