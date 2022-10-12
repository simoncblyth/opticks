#include <cassert>
#include <iomanip>
#include "PLOG.hh"
#include "SPath.hh"
#include "SCurandState.hh"
#include "SStr.hh"

const plog::Severity SCurandState::LEVEL = SLOG::EnvLevel("SCurandState", "DEBUG" );  
const char* SCurandState::RNGDIR = SPath::Resolve("$RNGDir", DIRPATH ) ; 
const char* SCurandState::NAME_PREFIX = "QCurandState" ; 
const char* SCurandState::DEFAULT_PATH   = SPath::Resolve("$RNGDir/QCurandState_1000000_0_0.bin", 0) ; 

std::string SCurandState::Stem(unsigned long long num, unsigned long long seed, unsigned long long offset)
{
    std::stringstream ss ; 
    ss << NAME_PREFIX << "_" << num << "_" << seed << "_" << offset  ;   
    std::string s = ss.str(); 
    return s ;   
} 
std::string SCurandState::Path(unsigned long long num, unsigned long long seed, unsigned long long offset)
{
    std::stringstream ss ; 
    ss << RNGDIR << "/" << Stem(num, seed, offset) << ".bin" ; 
    std::string s = ss.str(); 
    return s ;   
}

/**
SCurandState::GetRngMax
--------------------------

Find that file_size is not a mutiple of item content. 
Presumably the 44 bytes of content get padded to 48 bytes
in the curandState which is typedef to curandStateXORWOW.

**/

long SCurandState::GetRngMax(const char* path)
{
    FILE *fp = fopen(path,"rb");

    bool failed = fp == nullptr ;
    LOG_IF(fatal, failed ) << " unabled to open file [" << path << "]" ;
    assert(!failed);

    fseek(fp, 0L, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);

    long content_size = 44 ;
    long rngmax = file_size/content_size ;

    LOG(LEVEL)
        << " path " << path
        << " file_size " << file_size
        << " content_size " << content_size
        << " rngmax " << rngmax
        ;

    assert( file_size % content_size == 0 );
    return rngmax ; 
}






SCurandState::SCurandState(unsigned long long num_, unsigned long long seed_, unsigned long long offset_)
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

SCurandState::SCurandState(const char* spec_)
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

void SCurandState::init()
{
    if(spec)
    {
        std::vector<int> ivec ; 
        SStr::ISplit(spec, ivec, ':' ); 
        unsigned num_vals = ivec.size(); 
        assert( num_vals > 0 && num_vals <= 3 ); 

        num    =  num_vals > 0 ? ivec[0] : 1 ; 
        seed   =  num_vals > 1 ? ivec[1] : 0 ; 
        offset =  num_vals > 2 ? ivec[2] : 0 ; 

        if(num <= 100) num *= 1000000 ; // num <= 100 assumed to be in millions  
    }

    path = Path(num, seed, offset); 
    exists = SPath::Exists(path.c_str()); 
    rngmax = exists ? GetRngMax( path.c_str() ) : 0 ; 
}

std::string SCurandState::desc() const 
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





