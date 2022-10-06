#pragma once
/**
SLaunchSequence
================

This is an updated version of the old cudarap/LaunchSequence

Old defaults, chosen while using macOS mobile GPU Geforce 750M:: 

    unsigned max_blocks=128
    unsigned threads_per_block=256  
 
Example of a CUDA launch using this::

    init_rng<<<launch.blocks_per_launch, launch.threads_per_block>>>( launch.threads_per_launch, launch.thread_offset, dev_rng_states_launch, seed, offset );

Can experiment with envvars

    THREADS_PER_BLOCK
        rather constrained even with TITAN V,  TITAN RTX cannot exceed 1024   

    MAX_BLOCKS
        not constrained, the maximum is enormous 

**/


#include <vector>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <sstream>
#include <iomanip>
#endif 

#include "ssys.h"

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SLaunch 
{
   unsigned thread_offset ; 
   unsigned threads_per_launch ; 
   unsigned blocks_per_launch ; 
   unsigned threads_per_block ; 
   unsigned sequence_index ; 
   float    kernel_time ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   std::string desc() const ; 
#endif

}; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline std::string SLaunch::desc() const 
{
   std::stringstream ss ; 
   ss 
       << "SLaunch::desc"
       << " sequence_index " << std::setw(3) << sequence_index 
       << " thread_offset " << std::setw(7) << thread_offset
       << " threads_per_launch " << std::setw(6) << threads_per_launch
       << " blocks_per_launch " << std::setw(6) << blocks_per_launch
       << " threads_per_block " << std::setw(6) << threads_per_block 
       << " kernel_time (ms) " << std::setw(20) << std::fixed << std::setprecision(4) << kernel_time 
       ;

   std::string s = ss.str(); 
   return s ; 
}
#endif


struct SYSRAP_API SLaunchSequence 
{
    unsigned items ; 
    unsigned threads_per_block ;
    unsigned max_blocks ;

    std::vector<SLaunch> launches ;

    SLaunchSequence(unsigned items); 
    void init(); 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string brief() const ; 
    std::string desc() const ; 
    float total_time() const ; 
    unsigned total_threads() const ; 
#endif
}; 

inline SLaunchSequence::SLaunchSequence(unsigned items_)
    :
    items(items_),
    threads_per_block(ssys::getenvint("THREADS_PER_BLOCK", 512)),
    max_blocks(ssys::getenvint("MAX_BLOCKS", 128))
{
    init(); 
}

inline void SLaunchSequence::init()
{
    assert( threads_per_block <= 1024 ); // THREADS_PER_BLOCK is highly constrained, unlike MAX_BLOCKS

    launches.clear();
    unsigned thread_offset = 0 ;
    unsigned sequence_index = 0 ;

    while( thread_offset < items )
    {
        unsigned remaining = items - thread_offset ;
        unsigned blocks_per_launch = remaining / threads_per_block ;
        if(remaining % threads_per_block != 0) blocks_per_launch += 1 ;  
        if( blocks_per_launch > max_blocks ) blocks_per_launch = max_blocks ; 
        // blocks_per_launch sticks at max_blocks until the last launch of the sequence  

        unsigned threads_per_launch = blocks_per_launch * threads_per_block ; 
        if(threads_per_launch > remaining) threads_per_launch = remaining ;

        launches.push_back( { thread_offset, threads_per_launch, blocks_per_launch, threads_per_block, sequence_index, -1.f } );

        thread_offset += threads_per_launch ; 
        sequence_index += 1 ;
    }
}


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

inline float SLaunchSequence::total_time() const 
{
   float total = 0.0f ; 
   for(unsigned i=0 ; i < launches.size() ; i++ )
   {
       const SLaunch& launch = launches[i] ;
       if(launch.kernel_time > 0.f ) total += launch.kernel_time ;
   }
   return total ; 
}

inline unsigned SLaunchSequence::total_threads() const 
{
   unsigned total = 0 ; 
   for(unsigned i=0 ; i < launches.size() ; i++ )
   {
       const SLaunch& launch = launches[i] ;
       total += launch.threads_per_launch ;
   }
   return total ; 
}


inline std::string SLaunchSequence::brief() const
{
    std::stringstream ss ; 
    ss 
       << " items " << std::setw(7) << items
       << " total_threads " << std::setw(7) << total_threads()
       << " THREADS_PER_BLOCK " << std::setw(5) << threads_per_block
       << " MAX_BLOCKS " << std::setw(6) << max_blocks
       << " num_launches " << std::setw(4) << launches.size() 
       << " total_time " << std::setw(10) << std::fixed << std::setprecision(4) << total_time() 
       ;
    std::string s = ss.str(); 
    return s ;  
}

inline std::string SLaunchSequence::desc() const
{
    std::stringstream ss ;
    ss
        << "SLaunchSequence::desc"
        << brief()
        << std::endl 
        ;
    for(unsigned i=0 ; i < launches.size() ; i++)  
    {
        const SLaunch& launch = launches[i] ; 
        ss << launch.desc() << std::endl ; 
    } 
    std::string s = ss.str(); 
    return s ;  
}

#endif

