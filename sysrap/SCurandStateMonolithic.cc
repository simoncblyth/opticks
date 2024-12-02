#include <cassert>
#include <iomanip>
#include "PLOG.hh"

#include "spath.h"
#include "sdirectory.h"
#include "sstr.h"

#include "SCurandStateMonolithic.hh"
#include "SEventConfig.hh"

const plog::Severity SCurandStateMonolithic::LEVEL = SLOG::EnvLevel("SCurandStateMonolithic", "DEBUG" );  
const char* SCurandStateMonolithic::RNGDIR = spath::Resolve("${RNGDir:-$HOME/.opticks/rngcache/RNG}") ; 

//const char* SCurandStateMonolithic::NAME_PREFIX = "QCurandState" ; 
const char* SCurandStateMonolithic::NAME_PREFIX = "QCurandStateMonolithic" ; 
const char* SCurandStateMonolithic::DEFAULT_PATH = nullptr ; 

std::string SCurandStateMonolithic::Desc()  // static
{
    const char* path = Path() ; 
    long rngmax = RngMax() ; 

    std::stringstream ss ; 
    ss << "SCurandStateMonolithic::Desc" << std::endl 
       << " SEventConfig::MaxCurandState() " << SEventConfig::MaxCurandState() << std::endl
       << " SCurandStateMonolithic::Path() " << path << std::endl 
       << " SCurandStateMonolithic::RngMax() " << rngmax << std::endl 
       << " RNGDIR  " << RNGDIR << std::endl 
       ;
    std::string s = ss.str() ;
    return s ; 
}


/**
SCurandStateMonolithic::Path
-----------------------------

For concatenated loading this needs to become 
a directory path not a file path

**/

const char* SCurandStateMonolithic::Path()  // static
{
    if(DEFAULT_PATH == nullptr)
    {
        int rngmax = SEventConfig::MaxCurandState(); 
        int seed = 0 ; 
        int offset = 0 ; 
        // seed and offset could some from SEventConfig too 
        
        std::string path_ = Path_(rngmax, seed, offset ); 
        const char* path = path_.c_str() ;
        sdirectory::MakeDirsForFile(path); 
        DEFAULT_PATH = strdup(path); 
    }
    return DEFAULT_PATH ; 
}

std::string SCurandStateMonolithic::Stem_(ULL num, ULL seed, ULL offset)
{
    assert( num % M == 0 ); 
    ULL value = num/M ; 
    std::stringstream ss ; 
    ss << NAME_PREFIX << "_" << value << "M" << "_" << seed << "_" << offset  ;   
    std::string s = ss.str(); 
    return s ;   
} 
std::string SCurandStateMonolithic::Path_(ULL num, ULL seed, ULL offset)
{
    std::stringstream ss ; 
    ss << RNGDIR << "/" << Stem_(num, seed, offset) << ".bin" ; 
    std::string s = ss.str(); 
    return s ;   
}

/**
SCurandStateMonolithic::RngMax
-------------------------------

Determine *RngMax* based on file size of the 
configure curandstate array file divided by item_size of 44 bytes. 

Find that file_size is not a mutiple of item content (ie not sizeof(curandState))
Presumably the 44 bytes of content get padded to 48 bytes
in the curandState which is typedef to curandStateXORWOW.

**/
long SCurandStateMonolithic::RngMax()
{
    const char* path = Path(); 
    return RngMax(path) ; 
}

/**


generalizing this, path becomes a directory 
and the RngMax is determined by summing over
all files in the directory with the expected prefix

**/


long SCurandStateMonolithic::RngMax(const char* path)
{
    long file_size = spath::Filesize(path); 

    long item_size = 44 ;
    bool expected_file_size = file_size % item_size == 0 ; 
    long rngmax = file_size/item_size ;

    LOG(LEVEL)
        << " path " << path
        << " file_size " << file_size
        << " item_size " << item_size
        << " rngmax " << rngmax
        << " expected_file_size " << ( expected_file_size ? "YES" : "NO" )
        ;

    LOG_IF(fatal, !expected_file_size) << " NOT EXPECTED FILE SIZE " ; 
    assert( expected_file_size );
    return rngmax ; 
}




SCurandStateMonolithic::SCurandStateMonolithic(ULL num_, ULL seed_, ULL offset_)
    :
    spec(nullptr),
    num(num_),
    seed(seed_),
    offset(offset_),
    path(""),
    exists(false),
    rngmax(0)
{
    init(); 
}

SCurandStateMonolithic::SCurandStateMonolithic(const char* spec_)
    :
    spec(spec_ ? strdup(spec_) : nullptr),
    num(0),
    seed(0),
    offset(0),
    path(""),
    exists(false),
    rngmax(0)
{
    init(); 
}

void SCurandStateMonolithic::init()
{
    if(spec)
    {
        std::vector<int> ivec ; 
        sstr::split<int>(ivec, spec, ':'); 
        unsigned num_vals = ivec.size(); 
        assert( num_vals > 0 && num_vals <= 3 ); 

        num    =  num_vals > 0 ? ivec[0] : 1 ; 
        seed   =  num_vals > 1 ? ivec[1] : 0 ; 
        offset =  num_vals > 2 ? ivec[2] : 0 ; 

        if(num <= 100) num *= 1000000 ; // num <= 100 assumed to be in millions  
    }

    path = Path_(num, seed, offset); 
    exists = spath::Exists(path.c_str()); 
    rngmax = exists ? RngMax( path.c_str() ) : 0 ; 
}

std::string SCurandStateMonolithic::desc() const 
{
    std::stringstream ss ; 
    ss 
         << " spec " << std::setw(10) << ( spec ? spec : "-" ) 
         << " num " << std::setw(10) << num 
         << " seed " << std::setw(10) << seed 
         << " offset " << std::setw(10) << offset
         << " path " << std::setw(60) << path 
         << " exists " << exists
         << " rngmax " << rngmax 
         ; 
    std::string s = ss.str() ;
    return s ; 
}



