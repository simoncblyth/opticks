#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cstdlib>
#include <vector>
#include <string>
#endif

/**
SCSGPrimSpec :
=========================================================================

This was migrated down from CSG/CSGPrimSpec.h 

* *SCSGPrimSpec* provides the specification to access the AABB and sbtIndexOffset of all CSGPrim of a CSGSolid.  
* The specification includes pointers, counts and stride.
* Instances are created for a solidIdx by CSGFoundry::getPrimSpec using CSGPrim::MakeSpec
* Crucially *SCSGPrimSpec* is used to pass the AABB for a solid to CSGOptix/GAS_Builder.

Previously assumed that the *sbtIndexOffset* indices were global 
to the entire geometry, but the 2nd-GAS-last-prim-only bug indicates 
that the indices need to be local to each GAS, counting 
from 0 to numPrim-1 for that GAS.

**/

struct SCSGPrimSpec
{
    static constexpr const unsigned CSGPrim__value_offsetof_sbtIndexOffset = 4 ;
    static constexpr const unsigned CSGPrim__value_offsetof_AABB = 8 ; 
  
    const float*    aabb ; 
    const unsigned* sbtIndexOffset ;   
    unsigned        num_prim ; 
    unsigned        stride_in_bytes ; 
    bool            device ; 
    unsigned        primitiveIndexOffset ;   // offsets optixGetPrimitiveIndex() see GAS_Builder::MakeCustomPrimitivesBI_11N

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void downloadDump(const char* msg="SCSGPrimSpec::downloadDump") const ; 
    void gather(std::vector<float>& out) const ;
    static void Dump(std::vector<float>& out);
    void dump(const char* msg="SCSGPrimSpec::dump", int modulo=100) const ; 
    std::string desc() const ; 
#endif
};



#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>

#include "scuda.h"

#ifdef WITH_CUDA
#include "SCU.h"
#endif

/**
SCSGPrimSpec::gather
---------------------

Writes num_prim*6 bbox floats into out. 

**/

inline void SCSGPrimSpec::gather(std::vector<float>& out) const 
{
    //assert( device == false ); 
    unsigned size_in_floats = 6 ; 
    out.resize( num_prim*size_in_floats ); 

    unsigned stride_in_floats = stride_in_bytes/sizeof(float) ; 
    for(unsigned i=0 ; i < num_prim ; i++) 
    {   
        float* dst = out.data() + size_in_floats*i ;   
        const float* src = aabb + stride_in_floats*i ;   
        memcpy(dst, src,  sizeof(float)*size_in_floats );  
    }   
}

inline void SCSGPrimSpec::Dump(std::vector<float>& out)  // static 
{
     std::cout << " gather " << out.size() << std::endl ; 
     for(unsigned i=0 ; i < out.size() ; i++) 
     {    
         if(i % 6 == 0) std::cout << std::endl ; 
         std::cout << std::setw(10) << out[i] << " " ; 
     } 
     std::cout << std::endl ; 
}


inline void SCSGPrimSpec::dump(const char* msg, int modulo) const 
{
    assert( stride_in_bytes % sizeof(float) == 0 ); 
    unsigned stride_in_floats = stride_in_bytes/sizeof(float) ; 
    std::cout 
        << msg 
        << " num_prim " << num_prim 
        << " stride_in_bytes " << stride_in_bytes 
        << " stride_in_floats " << stride_in_floats 
        << " modulo " << modulo
        << std::endl 
        ; 

    for(unsigned i=0 ; i < num_prim ; i++)
    {   
        if( modulo == 0 || ( modulo > 0 && i % modulo == 0 ) )
        {
            std::cout 
                << " i " << std::setw(4) << i 
                << " sbtIndexOffset " << std::setw(4) << *(sbtIndexOffset + i*stride_in_floats)   
                ; 
            for(unsigned j=0 ; j < 6 ; j++)  
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << *(aabb + i*stride_in_floats + j ) << " "  ;   
            std::cout << std::endl ; 
        }
    }   
}


inline std::string SCSGPrimSpec::desc() const 
{
    std::stringstream ss ; 

    ss << "SCSGPrimSpec"
       << " primitiveIndexOffset " << std::setw(4) << primitiveIndexOffset
       << " num_prim " << std::setw(4) << num_prim 
       << " stride_in_bytes " << std::setw(5) << stride_in_bytes 
       << " device " << std::setw(2) << device
       ;

    std::string s = ss.str(); 
    return s ; 
}


/**
SCSGPrimSpec::downloadDump
---------------------------

As are starting the read from within the structure, 
need to trim offsets to avoid reading beyond the array.

**/

inline void SCSGPrimSpec::downloadDump(const char* msg) const 
{
#ifdef WITH_CUDA
    assert( device == true ); 
    unsigned stride_in_values = stride_in_bytes/sizeof(float) ; 
    unsigned numValues = stride_in_values*num_prim ;   
    unsigned nff = numValues - CSGPrim__value_offsetof_AABB ;  
    unsigned nuu = numValues - CSGPrim__value_offsetof_sbtIndexOffset ;
    std::cout 
        << "[ " << msg 
        << " num_prim " << num_prim 
        << " stride_in_values " << stride_in_values
        << " numValues " << numValues
        << " nff " << nff
        << " nuu " << nuu
        << " CSGPrim__value_offsetof_AABB " << CSGPrim__value_offsetof_AABB
        << " CSGPrim__value_offsetof_sbtIndexOffset " << CSGPrim__value_offsetof_sbtIndexOffset
        << std::endl 
        ; 

    assert( stride_in_values == 16 ); 

    std::vector<unsigned> uu ; 
    std::vector<float> ff ; 

    SCU::DownloadVec(ff,           aabb, nff); 
    SCU::DownloadVec(uu, sbtIndexOffset, nuu ); 

    for(unsigned i=0 ; i < num_prim ; i++)
    {
        std::cout << std::setw(5) << *(uu.data()+stride_in_values*i + 0) << " " ;
        for(int j=0 ; j < 6 ; j++) 
        {
            std::cout 
                << std::fixed << std::setw(8) << std::setprecision(2) 
                << *(ff.data()+stride_in_values*i + j) 
                << " " 
                ;
        } 
        std::cout << std::endl ; 
    }
    std::cout << "] " << msg << std::endl ; 
#else
    std::cout << "SCSGPrimSpec::downloadDump FATAL requires compilation WITH_CUDA \n" ;
    std::exit(1);
#endif
}


#endif


