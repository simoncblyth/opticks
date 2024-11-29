#pragma once
/**
SCurandState.h : More Flexible+capable replacement for SCurandState.{hh,cc}
============================================================================

Old impl fixes num (the total number of photon slots), 
instead of doing that fix *states_per_chunk* the number of photons in a chunk 
(maybe 1M photons per chunk in my usage, but it will be an input 
to SCurandState ctor)

The motivation is to allow the maxphoton to be dynamically decided 
by controlling the number of chunks in the available sequence to 
be concatenate loaded at runtime. 
 
File names will need additional metadata:

* states_per_chunk  (NOT strictly needed as can determine from filesize) 
* chunk_sequence_index
* NOTE: want to be extensible, so no chunk_total 


Another consideration is creation of the states, do no want to
need huge amounts of CPU memory to do that.


* TODO: require num to be in millions and change num to be expressed in millions with "M" suffix
* TODO: handle chunking by encoding (states_per_chunk, chunk_sequence_index) in the name
* TODO: use fixed chunk size 

  * hmm i recall doing something similar somewhere else with directory naming (probably precooked randoms)
  * want approach to be extendable, adding more chunks without recreating all of them  
 

NP::Load
   loads and concatenates arrays from a directory, will need similar : but cannot reuse
   as need to analyse the file names to get the right sequence and load+concat the 
   desired number of chunks 

   will need to enforce a contiguous set of chunk indices 


**/


#include "spath.h"

struct _SCurandState   
{
    static constexpr const long STATE_SIZE = 44 ;  
    static constexpr const char* PREFIX = "SCurandState_" ; 
    static constexpr const char* RNGDIR = "${RNGDir:-$HOME/.opticks/rngcache/RNG}" ; 
    static long FileStates(const char* name); 
    static long FileStates(const char* dir, const char* name); 
}; 


inline long _SCurandState::FileStates(const char* name)
{
    return FileStates(nullptr, name);
}
inline long _SCurandState::FileStates(const char* _dir, const char* name)
{
    const char* dir = _dir ? _dir : RNGDIR ; 
    long file_size = spath::Filesize(dir, name); 

    bool expected_file_size = file_size % STATE_SIZE == 0 ; 
    long states = file_size/STATE_SIZE ;

    std::cerr
        << "_SCurandState::FileStates"
        << " dir " << ( dir ? dir : "-" )
        << " name " << ( name ? name : "-" )
        << " file_size " << file_size
        << " STATE_SIZE " << STATE_SIZE
        << " states " << states
        << " expected_file_size " << ( expected_file_size ? "YES" : "NO" )
        << "\n"
        ;

    assert( expected_file_size );
    return states ; 
}


