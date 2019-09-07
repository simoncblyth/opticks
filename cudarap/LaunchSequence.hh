/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "CUDARAP_API_EXPORT.hh"

struct CUDARAP_API Launch {
   Launch(unsigned int _thread_offset, 
          unsigned int _threads_per_launch, 
          unsigned int _blocks_per_launch, 
          unsigned int _threads_per_block,
          unsigned int _sequence_index 
         )
          : 
          thread_offset(_thread_offset), 
          threads_per_launch(_threads_per_launch), 
          blocks_per_launch(_blocks_per_launch),
          threads_per_block(_threads_per_block), 
          sequence_index(_sequence_index),
          kernel_time(-1.0f)
          {}


   void Summary(const char* msg) const
   {
       printf(" %s sequence_index %3d  thread_offset %7u  threads_per_launch %6u blocks_per_launch %6u   threads_per_block %6u  kernel_time %10.4f ms \n", 
           msg ? msg : "-" ,
           sequence_index,
           thread_offset, 
           threads_per_launch,
           blocks_per_launch,
           threads_per_block, 
           kernel_time
           );
   } 

   unsigned int thread_offset ; 
   unsigned int threads_per_launch ; 
   unsigned int blocks_per_launch ; 
   unsigned int threads_per_block ; 
   unsigned int sequence_index ; 
   float        kernel_time ; 
}; 


/**
LaunchSequence
===============

Old defaults, chosen while using macOS mobile GPU Geforce 750M:: 

    unsigned max_blocks=128
    unsigned threads_per_block=256  
 
Example of a CUDA launch using this::

    init_rng<<<launch.blocks_per_launch, launch.threads_per_block>>>( launch.threads_per_launch, launch.thread_offset, dev_rng_states_launch, seed, offset );

Can experimeny with envvars

    THREADS_PER_BLOCK
        rather constrained even with TITAN V,  TITAN RTX cannot exceed 1024   

    MAX_BLOCKS
        not constrained, the maximum is enormous 


**/

class CUDARAP_API LaunchSequence {
public:
    LaunchSequence( unsigned int items, unsigned int threads_per_block , unsigned int max_blocks, bool reverse=false) : 
        m_items(items),
        m_threads_per_block(threads_per_block),
        m_max_blocks(max_blocks),
        m_reverse(reverse),
        m_tag(0)
    {
        update();
    }

private:
    void update()
    {
        m_launches.clear();
        unsigned int thread_offset = 0 ;
        unsigned int sequence_index = 0 ;

        while( thread_offset < m_items )
        {
             unsigned int remaining = m_items - thread_offset ;

             unsigned int blocks_per_launch = remaining / m_threads_per_block ;
             if(remaining % m_threads_per_block != 0) blocks_per_launch += 1 ;  
             if( blocks_per_launch > m_max_blocks ) blocks_per_launch = m_max_blocks ; 
             // blocks_per_launch sticks at m_max_blocks until the last launch of the sequence  

             unsigned int threads_per_launch = blocks_per_launch * m_threads_per_block ; 
             if(threads_per_launch > remaining) threads_per_launch = remaining ;

             m_launches.push_back(Launch(thread_offset, threads_per_launch, blocks_per_launch, m_threads_per_block, sequence_index));

             thread_offset += threads_per_launch ; 
             sequence_index += 1 ;
        }
    }

public:
    LaunchSequence* copy(unsigned int max_blocks=0, unsigned int threads_per_block=0)
    {
       return new LaunchSequence(
                     m_items, 
                     threads_per_block > 0 ? threads_per_block : m_threads_per_block, 
                     max_blocks        > 0 ? max_blocks        : m_max_blocks,
                     m_reverse
                    ); 
    }

    unsigned int getNumLaunches(){     return m_launches.size(); }
    unsigned int getItems(){           return m_items ; }
    unsigned int getThreadsPerBlock(){ return m_threads_per_block ; }
    unsigned int getMaxBlocks(){       return m_max_blocks ; }
    char*        getTag(){             return m_tag ; }
    bool         getReverse(){         return m_reverse ; }

    void setTag(const char* tag)
    {
        m_tag = strdup(tag) ;
    }

    void setMaxBlocks(unsigned int max_blocks)
    {
        m_max_blocks = max_blocks ;
        update();
    }

    void setThreadsPerBlock(unsigned int threads_per_block)
    {
        m_threads_per_block = threads_per_block ;
        update();
    }

    void setItems(unsigned int items)
    {
        m_items = items ;
        update();
    }




    virtual ~LaunchSequence()
    {
        free(m_tag);
    }


    float getTotalTime(){
        float total = 0.0f ; 
        for(unsigned int i=0 ; i<getNumLaunches() ; i++ )
        {
            Launch& launch = getLaunch(i) ;
            if(launch.kernel_time > 0.f ) total += launch.kernel_time ;
        }
        return total ; 
    } 

    Launch& getLaunch(unsigned int i)
    { 
         unsigned int nlaunch = m_launches.size() ; 
         return m_reverse ? m_launches[nlaunch - 1 - i] : m_launches[i] ; 
    }

    void Summary(const char* msg)
    {
        unsigned int nlaunch = getNumLaunches();
        printf("%s tag %s workitems %7u  threads_per_block %5u  max_blocks %6u reverse %1d nlaunch %3u TotalTime %10.4f ms \n", 
            msg ? msg : "" ,
            m_tag ? m_tag : "",
            m_items,
            m_threads_per_block,
            m_max_blocks,
            m_reverse,
            nlaunch,
            getTotalTime() 
        ); 

    }

    void dump(const char* msg)  
    {
        Summary(msg);
        unsigned int nlaunch = getNumLaunches();
        for(unsigned int i=0 ; i<nlaunch ; i++ ) getLaunch(i).Summary(msg);
    }


private:
    unsigned int m_items ; 
    unsigned int m_threads_per_block ;
    unsigned int m_max_blocks ;
    bool         m_reverse ;

private:
    std::vector<Launch> m_launches ;
    char* m_tag ;

};



