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
#include "Shape.h"
#include "Binding.h"
#include "Params.h"

#include "GAS.h"
#include "GAS_Builder.h"

#include "IAS.h"
#include "IAS_Builder.h"

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
    createGAS(geo); 
    createIAS(geo); 
    setTop(geo->top); 

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



unsigned SBT::getNumBI() const // total 
{
    unsigned tot = 0 ; 
    for(unsigned i=0 ; i < nbis.size() ; i++) tot += nbis[i] ; 
    return tot ; 
}
unsigned SBT::getNumBI(unsigned shape_idx) const 
{
    assert( shape_idx < nbis.size()); 
    return nbis[shape_idx] ; 
}
void SBT::dumpOffsetBI() const 
{
    unsigned num_shape = vgas.size() ; 
    std::cout << "SBT::dumpOffsetBI  num_shape " << num_shape << std::endl ; 
    for(unsigned shape_idx=0 ; shape_idx < num_shape ; shape_idx++)
    {
        unsigned num_bi = getNumBI(shape_idx); 
        unsigned offset_bi = getOffsetBI(shape_idx); 
        std::cout 
            << " shape_idx " << std::setw(6) << shape_idx  
            << " num_bi " << std::setw(6) << num_bi
            << " offset_bi " << std::setw(6) << offset_bi
            << std::endl
            ;
    }
}



unsigned SBT::getOffsetBI(unsigned shape_idx) const 
{
    assert( shape_idx < nbis.size()); 
    unsigned offset = 0 ; 
    for(unsigned i=0 ; i < nbis.size() ; i++) 
    {
        if( i == shape_idx ) break ; 
        offset += nbis[i]; 
    }
    return offset ;     
} 


/**
Geo::getOffsetBI
------------------

Example::

    GAS_0            --> 0 
        BI_0 
        BI_1
    GAS_1            --> 2 
        BI_0 
        BI_1
    GAS_2            --> 4 
        BI_0 

Simply by keeping count of build inputs (BI) used for each GAS.

**/



void SBT::createGAS(const Geo* geo)
{
    unsigned num_shape = geo->getNumShape(); 
    for(unsigned i=0 ; i < num_shape ; i++)
    {
        const Shape* sh = geo->getShape(i) ;    
        GAS gas = {} ;  
        GAS_Builder::Build(gas, sh );
        vgas.push_back(gas);  
        unsigned num_bi = gas.bis.size() ;
        nbis.push_back(num_bi); 
    }
}

void SBT::createIAS(const Geo* geo)
{
    unsigned num_grid = geo->getNumGrid(); 
    for(unsigned i=0 ; i < num_grid ; i++)
    {
        const Grid* gr = geo->getGrid(i) ;    
        IAS ias = {} ;  
        IAS_Builder::Build(ias, gr, this );
        vias.push_back(ias);  
    }
}

const GAS& SBT::getGAS(unsigned gas_idx) const 
{
    assert( gas_idx < vgas.size()); 
    return vgas[gas_idx]; 
}

const IAS& SBT::getIAS(unsigned ias_idx) const 
{
    assert( ias_idx < vias.size()); 
    return vias[ias_idx]; 
}


AS* SBT::getTop() const 
{
    return top ; 
}

void SBT::setTop(const char* spec)
{
    AS* a = getAS(spec); 
    setTop(a); 
}
void SBT::setTop(AS* top_)
{   
    top = top_ ;
}

AS* SBT::getAS(const char* spec) const 
{
   assert( strlen(spec) > 1 );  
   char c = spec[0]; 
   assert( c == 'i' || c == 'g' );  
   int idx = atoi( spec + 1 );  

   std::cout << "SBT::getAS " << spec << " c " << c << " idx " << idx << std::endl ; 

   AS* a = nullptr ; 
   if( c == 'i' )
   {   
       const IAS& ias = vias[idx]; 
       a = (AS*)&ias ; 
   }   
   else if( c == 'g' )
   {   
       const GAS& gas = vgas[idx]; 
       a = (AS*)&gas ; 
   }   

   if(a)
   {   
       std::cout << "SBT::getAS " << spec << std::endl ; 
   }   
   return a ; 
}





void SBT::createHitgroup(const Geo* geo)
{
    unsigned num_shape = geo->getNumShape(); 
    unsigned num_gas = vgas.size(); 
    assert( num_gas == num_shape ); 

    unsigned tot_bi = 0 ; 
    for(unsigned i=0 ; i < num_gas ; i++)
    {
        const GAS& gas = vgas[i] ;    
        tot_bi += gas.bis.size() ;  
    }
    assert( tot_bi > 0 );  
    std::cout 
        << "SBT::createHitgroup"
        << " num_shape " << num_shape 
        << " num_gas " << num_gas 
        << " tot_bi " << tot_bi 
        << std::endl 
        ; 

    hitgroup = new HitGroup[tot_bi] ; 
    HitGroup* hg = hitgroup ; 
    unsigned  tot_bi2 = 0 ; 

    for(unsigned i=0 ; i < tot_bi ; i++)   // pack headers CPU side
         OPTIX_CHECK( optixSbtRecordPackHeader( pip->hitgroup_pg, hitgroup + i ) ); 
    
    for(unsigned i=0 ; i < num_gas ; i++)
    {
        const GAS& gas = vgas[i] ;    
        unsigned num_bi = gas.bis.size(); 
        const Shape* sh = gas.sh ; 
  
        std::cout << "SBT::createHitgroup gas_idx " << i << " num_bi " << num_bi << std::endl ; 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 

            //const float* aabb = sh->get_aabb(j) ; 
            //const float* param = sh->get_param(j) ; 
            // how to organize generalization by primitive OR CSG tree type ?
            float radius = sh->get_size(j); 

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



