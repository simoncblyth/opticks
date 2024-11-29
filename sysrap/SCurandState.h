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

Related
--------

qudarap/QCurandState.{hh,cc,cu}
   alloc+create+download+save

sysrap/SLaunchSequence.h
   configure launches

qudarap/QRng.{hh.cc}




**/


#include "sdirectory.h"
#include "spath.h"
#include <iomanip>

struct _SCurandChunk
{
    int idx ;     
    int num ; 
    int seed ;
    int offset ; 

    std::string format() const ;

    static constexpr const char* PREFIX = "SCurandChunk_" ; 
    static constexpr const char* EXT = ".bin" ; 
    static constexpr char DELIM = '_' ;
    static constexpr const int NUM_ELEM = 4 ;  

    static int ParseDir( std::vector<_SCurandChunk>& chunks, const char* _dir );
    static int ParseName( _SCurandChunk& chunk, const char* name ); 
    static int ParseNum(const char* num); 

    static std::string FormatIdx(int idx);
    static std::string FormatNum(int num); 
};


inline int _SCurandChunk::ParseDir(std::vector<_SCurandChunk>& chunks, const char* _dir )
{
    const char* dir = spath::Resolve(_dir) ; 
    std::vector<std::string> names ; 
    sdirectory::DirList( names, dir, PREFIX, EXT ); 

    int num_names = names.size(); 
    for(int i=0 ; i < num_names ; i++) 
    {
        const std::string& n = names[i] ; 
        _SCurandChunk c = {} ; 
        if(_SCurandChunk::ParseName(c, n.c_str())==0) chunks.push_back(c); 
    }
    return 0 ; 
}

inline int _SCurandChunk::ParseName( _SCurandChunk& chunk, const char* name )
{
    if(name == nullptr ) return 1 ;  
    size_t other = strlen(PREFIX)+strlen(EXT) ; 
    if( strlen(name) <= other ) return 2 ; 

    std::string n = name ; 
    std::string meta = n.substr( strlen(PREFIX), strlen(name) - other ); 


    std::vector<std::string> elem ; 
    sstr::Split(meta.c_str(), DELIM, elem ); 

    unsigned num_elem = elem.size(); 
    if( num_elem != NUM_ELEM )  return 3 ; 

    chunk.idx    =  std::atoi(elem[0].c_str()) ; 
    chunk.num    =  ParseNum(elem[1].c_str()) ; 
    chunk.seed   =  std::atoi(elem[2].c_str()) ; 
    chunk.offset =  std::atoi(elem[3].c_str()) ; 

    std::cout << "_SCurandChunk::ParseName " << std::setw(30) << n << " : [" << meta << "][" << chunk.format() << "]\n" ; 
    return 0 ; 
}

inline int _SCurandChunk::ParseNum(const char* num)
{
    char* n = strdup(num); 
    char last = n[strlen(n)-1] ; 
    int scale = 1 ; 
    switch(last)
    {
        case 'k': scale = 1000    ; break ;  
        case 'M': scale = 1000000 ; break ;  
    }
    if(scale > 1) n[strlen(n)-1] = '\0' ; 
    int value = scale*std::atoi(num) ; 
    return value ; 
}


inline std::string _SCurandChunk::format() const
{
    std::stringstream ss ; 
    ss << FormatIdx(idx) 
       << DELIM
       << FormatNum(num)
       << DELIM
       << seed
       << DELIM
       << offset
       ; 
    std::string str = ss.str(); 
    return str ;   
}
inline std::string _SCurandChunk::FormatIdx(int idx)
{
    std::stringstream ss; 
    ss << std::setw(4) << std::setfill('0') << idx ;
    std::string str = ss.str(); 
    return str ;   
}
inline std::string _SCurandChunk::FormatNum(int num)
{
    int scale = 1 ; 
    if( num >= 1000000 )     scale = 1000000 ;
    else if( scale >= 1000 ) scale = 1000 ;  
    assert( num % scale == 0 && "integer multiples of 1000 or 1000000 are required" ); 

    char suffix = '\0' ; 
    switch(scale)
    {
       case 1000:    suffix = 'k' ; break ;  
       case 1000000: suffix = 'M' ; break ;  
    }

    std::stringstream ss; 
    ss << num/scale ; 
    if( suffix != '\0' ) ss << suffix ; 

    std::string str = ss.str(); 
    return str ;   
}





struct _SCurandState   
{
    _SCurandState(const char* dir=nullptr); 

    std::vector<_SCurandChunk> chunks ; 

    static constexpr const long STATE_SIZE = 44 ;  

    static constexpr const char* RNGDIR = "${RNGDir:-$HOME/.opticks/rngcache/RNG}" ; 
    static long FileStates(const char* name); 
    static long FileStates(const char* dir, const char* name); 
}; 


inline _SCurandState::_SCurandState(const char* _dir)
{
    const char* dir = _dir ? _dir : RNGDIR ; 

    _SCurandChunk::ParseDir(chunks, dir);    // TODO: needs to work with non-existing dir too 
}


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



