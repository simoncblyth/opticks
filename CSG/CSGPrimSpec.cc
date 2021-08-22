

#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>

#include "scuda.h"

#include "CSGPrim.h"
#include "CSGPrimSpec.h"
#include "CU.h"


/**
CSGPrimSpec::gather
---------------------

Writes num_prim*6 bbox floats into out. 

**/

void CSGPrimSpec::gather(std::vector<float>& out) const 
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

void CSGPrimSpec::Dump(std::vector<float>& out)  // static 
{
     std::cout << " gather " << out.size() << std::endl ; 
     for(unsigned i=0 ; i < out.size() ; i++) 
     {    
         if(i % 6 == 0) std::cout << std::endl ; 
         std::cout << std::setw(10) << out[i] << " " ; 
     } 
     std::cout << std::endl ; 
}


void CSGPrimSpec::dump(const char* msg) const 
{
    assert( stride_in_bytes % sizeof(float) == 0 ); 
    unsigned stride_in_floats = stride_in_bytes/sizeof(float) ; 
    std::cout 
        << msg 
        << " num_prim " << num_prim 
        << " stride_in_bytes " << stride_in_bytes 
        << " stride_in_floats " << stride_in_floats 
        << std::endl 
        ; 

    for(unsigned i=0 ; i < num_prim ; i++)
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

/**
CSGPrimSpec::downloadDump
---------------------------

As are starting the read from within the structure, 
need to trim offsets to avoid reading beyond the array.

**/

void CSGPrimSpec::downloadDump(const char* msg) const 
{
    assert( device == true ); 
    unsigned stride_in_values = stride_in_bytes/sizeof(float) ; 
    unsigned numValues = stride_in_values*num_prim ;   
    unsigned nff = numValues - CSGPrim::value_offsetof_AABB() ;  
    unsigned nuu = numValues - CSGPrim::value_offsetof_sbtIndexOffset() ;
    std::cout 
         << "[ " << msg 
         << " num_prim " << num_prim 
         << " stride_in_values " << stride_in_values
         << " numValues " << numValues
         << " nff " << nff
         << " nuu " << nuu
         << " CSGPrim::value_offsetof_AABB " << CSGPrim::value_offsetof_AABB()
         << " CSGPrim::value_offsetof_sbtIndexOffset " << CSGPrim::value_offsetof_sbtIndexOffset()
         << std::endl 
         ; 

    assert( stride_in_values == 16 ); 

    std::vector<unsigned> uu ; 
    std::vector<float> ff ; 

    CU::DownloadVec(ff,           aabb, nff); 
    CU::DownloadVec(uu, sbtIndexOffset, nuu ); 

    for(int i=0 ; i < num_prim ; i++)
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
}


#endif

