#pragma once
/**
SCurandState.h : More Flexible+capable replacement for SCurandState.{hh,cc}
============================================================================

::

    ~/o/sysrap/tests/SCurandState_test.sh


Old impl fixes num (the total number of photon slots), 
instead of doing that move to a chunk-centric implementation
in a way that is extensible via addition of chunks 
and flexible by not loading all the chunks, or even partially 
loading some chunks to get the desired number of states.  

The motivation is to allow the maxphoton to be dynamically decided 
depending on VRAM by controlling the number of chunks in the available 
sequence to be loaded+uploaded at runtime. 
 
Actually there is no need for concatenation CPU side (only GPU side). 
Can load and upload chunk-by-chunk to complete the contiguous 
states GPU side. 


Related
--------

qudarap/QCurandState.{hh,cc,cu}
   alloc+create+download+save

sysrap/SLaunchSequence.h
   configure launches

qudarap/QRng.{hh.cc}

**/


#include "SCurandChunk.h"
#include "SCurandSize.h"
#include "SYSRAP_API_EXPORT.hh"


struct SYSRAP_API _SCurandState   
{
    typedef unsigned long long ULL ;
    const char* dir ; 
    scurandref all = {} ; 
    scurandref* d_all = nullptr ; 
 
    std::vector<ULL> size = {} ; 
    std::vector<SCurandChunk> chunk = {} ; 

    _SCurandState(const char* dir=nullptr); 
    void init(); 
    void initFromSize(); 
    void initMerged();

    void addChunk(ULL num);
 
    ULL num_total() const ;
    std::string desc() const ; 
 
    bool is_complete() const ; 
}; 


inline _SCurandState::_SCurandState(const char* _dir)
    :
    dir( _dir ? strdup(_dir) : nullptr )
{
    init(); 
}

inline void _SCurandState::init()
{
    all.chunk_idx = 0 ; 
    all.chunk_offset = 0 ; 

    all.num = 0 ;
    all.seed = 0 ; 
    all.offset = 0 ; 
    all.states = nullptr ; 

    SCurandSize::ParseSpec(size, nullptr); 
    SCurandChunk::ParseDir(chunk, dir);  

    int num_size = size.size() ; 
    int num_chunk = chunk.size() ; 

    if(num_chunk == 0)
    {
        initFromSize(); 
    }
    else if( num_chunk > 0 )
    {
        initMerged(); 
    }

    all.num = num_total(); 
}


inline void _SCurandState::initFromSize()
{
    long num_size = size.size() ; 
    for(long i=0 ; i < num_size ; i++) 
    {
       ULL num = size[i]; 
       addChunk(num);
    }
}


/**
_SCurandState::initMerged
----------------------------

Expecting to miss chunks at the end, not the beginning 
And existing chunks expected to be consistent with the 
sizes and indices that this code would have created. 
So changing the size SPEC for example would trip this up. 
**/

inline void _SCurandState::initMerged()
{
    long num_size = size.size() ; 
    for(long i=0 ; i < num_size ; i++)
    {
        scurandref* d = SCurandChunk::Find(chunk, i );
        if(d)
        {
            bool consistent = 
                      d->chunk_idx == i && 
                      d->num == size[i] &&
                      d->seed == all.seed &&
                      d->offset == all.offset 
                      ;  
            assert(consistent); 
            if(!consistent) return ;  
        } 
        else
        {
            addChunk(size[i]); 
        }
    }
}




inline void _SCurandState::addChunk(ULL num)
{
    int num_chunk = chunk.size(); 
    ULL num_cumulative = SCurandChunk::NumTotal_InRange(chunk, 0, num_chunk ); 

    SCurandChunk c = {} ; 

    c.ref.chunk_idx = num_chunk ; 
    c.ref.chunk_offset = num_cumulative ; 
    c.ref.num = num ; 
    c.ref.seed = all.seed ; 
    c.ref.offset = all.offset ; 
    c.ref.states = nullptr ; 

    chunk.push_back(c); 
}

inline unsigned long long _SCurandState::num_total() const
{
    return SCurandChunk::NumTotal_SizeCheck(chunk, size); 
}


inline std::string _SCurandState::desc() const 
{
    int num_size = size.size(); 
    int num_chunk = chunk.size(); 
    int num_valid = SCurandChunk::CountValid(chunk, dir );  
    bool complete = is_complete(); 

    std::stringstream ss; 
    ss 
       << "[_SCurandState::desc\n" 
       << " num_size " << num_size  << "\n" 
       << " num_chunk " << num_chunk << "\n" 
       << " num_valid " << num_valid << "\n" 
       << " is_complete " << ( complete ? "YES" : "NO " ) << "\n" 
       << " num_total " << SCurandChunk::FormatNum(num_total()) << "\n"
       << SCurandSize::Desc(size)  
       << "\n"  
       << SCurandChunk::Desc(chunk, dir) 
       << "]_SCurandState::desc\n" 
       ; 
    std::string str = ss.str(); 
    return str ;   
} 

inline bool _SCurandState::is_complete() const
{
    int num_size = size.size(); 
    int num_valid = SCurandChunk::CountValid(chunk, dir );  
    return num_size == num_valid ; 
} 

