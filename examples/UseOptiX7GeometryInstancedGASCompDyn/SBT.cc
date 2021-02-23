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
    unsigned num_rec = 0 ; 
    for(unsigned i=0 ; i < num_gas ; i++)
    {
        const GAS& gas = geo->getGAS(i) ;    
        num_rec += gas.num_sbt_rec ;  
    }
    assert( num_rec > 0 );  
    std::cout << "SBT::createHitgroup num_gas " << num_gas << " num_rec " << num_rec << std::endl ; 

    hitgroup = new HitGroup[num_rec] ; 
    HitGroup* hg = hitgroup ; 
    unsigned  hg_count = 0 ; 

    for(unsigned i=0 ; i < num_rec ; i++)   // pack headers CPU side
         OPTIX_CHECK( optixSbtRecordPackHeader( pip->hitgroup_pg, hitgroup + i ) ); 
    
    for(unsigned i=0 ; i < num_gas ; i++)
    {
        const GAS& gas = geo->getGAS(i) ;    
        const std::vector<float>& extents = gas.extents ; 
        unsigned num_sub = extents.size(); 
  
        std::cout << "SBT::createHitgroup gas_idx " << i << " num_sub " << num_sub << std::endl ; 

        for(unsigned j=0 ; j < num_sub ; j++)
        { 
            unsigned num_items = 1 ; 
            float* values = new float[num_items]; 
            values[0] = extents[j] ;  
            float* d_values = UploadArray<float>(values, num_items) ; 
            delete [] values ; 
      
            hg->data.values = d_values ; // set device pointer into CPU struct about to be copied to device
            hg++ ; 
            hg_count++ ; 
        }
    }
    assert( num_rec == hg_count );  

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup ), sizeof(HitGroup)*num_rec ) );
    sbt.hitgroupRecordBase  = d_hitgroup;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroup);
    sbt.hitgroupRecordCount = num_rec ;

    CUDA_CHECK( cudaMemcpy(   
                reinterpret_cast<void*>( d_hitgroup ),
                hitgroup,
                sizeof( HitGroup )*num_rec,
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
                reinterpret_cast<void*>( d_array ),
                sizeof(T)*num_items,
                cudaMemcpyDeviceToHost
                ) );

    return array ; 
}




template float* SBT::UploadArray<float>(const float* array, unsigned num_items) ;
template float* SBT::DownloadArray<float>(const float* d_array, unsigned num_items) ;



