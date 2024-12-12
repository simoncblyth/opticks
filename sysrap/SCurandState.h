#pragma once
/**
SCurandState.h : More Flexible+capable replacement for SCurandState.{hh,cc}
============================================================================

::

    ~/o/sysrap/tests/SCurandState_test.sh


Old SCurandState.{hh,cc} impl fixed num, the entire total number of photon slots. 
This SCurandState.h impl moves to a chunk-centric approach in order to be: 

1. extensible via addition of chunks 
2. flexible by not loading all chunks with partial loading 
   of the last chunk if necessary to meet the runtime configured 
   maximum number of states.  

The motivation is to allow the maxphoton to be dynamically decided 
depending on VRAM by controlling the number of chunks in the available 
sequence to be loaded+uploaded at runtime. 
 
Actually there is no need for concatenation CPU side (only GPU side). 
Can load and upload chunk-by-chunk to complete the contiguous 
states GPU side. 


Related
--------

qudarap/QCurandStateMonolithic.{hh,cc,cu}
   alloc+create+download+save

sysrap/SLaunchSequence.h
   configure launches

qudarap/QRng.{hh.cc}

**/


#include "SCurandChunk.h"
#include "SCurandSpec.h"
#include "SYSRAP_API_EXPORT.hh"
#include "SCU_.h"

struct SYSRAP_API SCurandState   
{
    typedef unsigned long long ULL ;
    static constexpr const ULL M = 1000000ull ; 

    const char* dir ; 
    std::vector<ULL> spec = {} ; 
    std::vector<SCurandChunk> chunk = {} ; 
    scurandref all = {} ; 

    const char* getDir() const ; 

    SCurandState(const char* dir=nullptr); 

    void init(); 
    void initFromSpec(); 
    void initFromDir();
    void addChunk(ULL num);
 
    ULL num_total() const ;
    std::string desc() const ; 
 
    bool is_complete() const ; 

    template<typename T>
    T* loadAndUpload( unsigned rngmax ) ; 

}; 


inline const char* SCurandState::getDir() const 
{
    return SCurandChunk::Dir(dir); 
}

inline SCurandState::SCurandState(const char* _dir)
    :
    dir( _dir ? strdup(_dir) : nullptr )
{
    init(); 
}


/**
SCurandState::init
---------------------

1. parse SPEC populating spec vector with slots per chunk values
2. parse directory populating chunk vector based on file names

Whether chunk files exist or not already 
the outcome of this instanciation remains the same, 
namely a chunk vector that follows the spec vector. 

**/

inline void SCurandState::init()
{
    all = {} ; 
    all.chunk_idx = 0 ; 
    all.chunk_offset = 0 ; 
    all.num = 0 ;
    all.states = nullptr ; 

    const char* SEED_OFFSET = ssys::getenvvar("SCurandState__init_SEED_OFFSET"); 
    const char* CHUNKSIZES  = ssys::getenvvar("SCurandState__init_CHUNKSIZES"); 

    SCurandSpec::ParseSeedOffset( all.seed, all.offset, SEED_OFFSET ); 
    SCurandSpec::ParseChunkSizes( spec, CHUNKSIZES  ); 

    SCurandChunk::ParseDir(chunk, dir);  

    int num_spec = spec.size() ; 
    int num_chunk = chunk.size() ; 

    if(num_chunk == 0)   // no chunk files found
    {
        initFromSpec(); 
    }
    else if( num_chunk > 0 )
    {
        initFromDir(); 
    }

    all.num = num_total(); 

    assert( spec.size() == chunk.size() );
}




inline void SCurandState::initFromSpec()
{
    long num_spec = spec.size() ; 
    for(long i=0 ; i < num_spec ; i++) 
    {
       ULL num = spec[i]; 
       addChunk(num);
    }
}


/**
SCurandState::initFromDir
----------------------------

Expecting to find the first chunks to be consistent
with the spec. When SPEC is extended appropriately 
then chunks will be missing from the end, and will be 
added from the spec. 

Changing the size SPEC is any way that does not just extend
the chunks would cause this to assert. 

HMM: in principal could arrange to replace missing chunks
but that luxury feature has not been implemented. 


* iteration over spec vector to support spec extension

* note that when the directory has all the chunk files
  already this does nothing other than checking that 
  the chunks follow the spec, the work of forming 
  the chunks being done in the init

**/

inline void SCurandState::initFromDir()
{
    ULL num_spec = spec.size() ; 

    for(ULL i=0 ; i < num_spec ; i++)
    {
        scurandref* r = SCurandChunk::Find(chunk, i );
        ULL num_cumulative = SCurandChunk::NumTotal_InRange(chunk, 0, i ); 
        bool already_have_chunk = r != nullptr ; 

        if(already_have_chunk)   
        {
            bool r_chunk_follows_spec = 
                      r->chunk_idx == i && 
                      r->chunk_offset == num_cumulative && 
                      r->num == spec[i] &&
                      r->seed == all.seed &&
                      r->offset == all.offset 
                      ;  

            assert(r_chunk_follows_spec); 
            if(!r_chunk_follows_spec) std::cerr
                << "SCurandState::initFromDir"
                << " r_chunk_follows_spec " << ( r_chunk_follows_spec ? "YES" : "NO " ) << "\n"
                << " r.chunk_idx " << r->chunk_idx << " i " << i << "\n" 
                << " r.chunk_offset " << r->chunk_offset << " num_cumulative " << num_cumulative << "\n" 
                << " r.num " << r->num << " spec[i] " << spec[i] << "\n"
                << " r.seed " << r->seed << " all.seed " << all.seed << "\n"
                << " r.offset " << r->offset << " all.offset " << all.offset << "\n"
                ; 

            if(!r_chunk_follows_spec) return ;  
        } 
        else
        {
            addChunk(spec[i]); 
        }
    }
}




inline void SCurandState::addChunk(ULL num)
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

inline unsigned long long SCurandState::num_total() const
{
    return SCurandChunk::NumTotal_SpecCheck(chunk, spec); 
}


inline std::string SCurandState::desc() const 
{
    const char* _dir = getDir();

    int num_spec = spec.size(); 
    int num_chunk = chunk.size(); 
    int num_valid = SCurandChunk::CountValid(chunk, dir );  
    bool complete = is_complete(); 

    std::stringstream ss; 
    ss 
       << "[SCurandState::desc\n" 
       << " dir " << ( dir ? dir : "-" ) << "\n" 
       << " getDir " << ( _dir ? _dir : "-" ) << "\n" 
       << " num_spec " << num_spec  << "\n" 
       << " num_chunk " << num_chunk << "\n" 
       << " num_valid " << num_valid << "\n" 
       << " is_complete " << ( complete ? "YES" : "NO " ) << "\n" 
       << " num_total " << SCurandChunk::FormatNum(num_total()) << "\n"
       << SCurandSpec::Desc(spec)  
       << "\n"  
       << SCurandChunk::Desc(chunk, dir) 
       << "]SCurandState::desc\n" 
       ; 
    std::string str = ss.str(); 
    return str ;   
} 

inline bool SCurandState::is_complete() const
{
    int num_spec = spec.size(); 
    int num_valid = SCurandChunk::CountValid(chunk, dir );  
    return num_spec == num_valid ; 
} 


/**
SCurandState::loadAndUpload
----------------------------

This is intended to replace QRng::LoadAndUpload 

**/


template<typename T>
inline T* SCurandState::loadAndUpload( unsigned rngmax )
{
    T* d0 = SCU_::device_alloc<T>( rngmax, "SCurandState::loadAndUpload" );
    T* d = d0 ; 

    ULL available_chunk = chunk.size(); 
    ULL count = 0 ; 

    std::cout
        << " rngmax " << rngmax
        << " rngmax/M " << rngmax/M
        << " available_chunk " << available_chunk 
        << " all.num/M " << all.num/M 
        << " rngmax/M " << rngmax/M
        << " d0 " << d0  
        << "\n"
        ;


    sdigest dig ; 

    for(ULL i=0 ; i < available_chunk ; i++)
    {   
        ULL remaining = rngmax - count ;   

        const SCurandChunk& ck = chunk[i]; 
 
        bool partial_read = remaining < ck.ref.num ;   

        ULL num = partial_read ? remaining : ck.ref.num ;

        std::cout 
            << " i " << std::setw(3) << i 
            << " ck.ref.num/M " << std::setw(4) << ck.ref.num/M
            << " count/M " << std::setw(4) << count/M
            << " remaining/M " << std::setw(4) << remaining/M
            << " partial_read " << ( partial_read ? "YES" : "NO " )
            << " num/M " << std::setw(4) << num/M
            << " d " << d 
            << "\n"
            ;

        scurandref cr = ck.load(num, dir, &dig ) ; 
  
        assert( cr.states != nullptr); 

        bool num_match = cr.num == num ; 

        if(!num_match) std::cerr
            << "SCurandState::loadAndUpload"
            << " num_match " << ( num_match ? "YES" : "NO " )
            << " cr.num/M " << cr.num/M
            << " num/M " << num/M
            << "\n"
            ;

        assert(num_match); 

        SCU_::copy_host_to_device<T>( d , cr.states , num );  

        free(cr.states); 

        d += num ;   
        count += num ;   

        if(count > rngmax) assert(0); 
        if(count == rngmax) break ;
    }   

    bool complete = count == rngmax ;
    assert( complete );
    std::string digest = dig.finalize();

    std::cout
        << "SCurandState::loadAndUpload"
        << " complete " << ( complete ? "YES" : "NO ")
        << " rngmax/M " << rngmax/M
        << " rngmax " << rngmax
        << " digest " << digest << "(cf md5sum of cat-ed chunk(s))" 
        << "\n"
        ;

    return complete ? d0 : nullptr ;
}



