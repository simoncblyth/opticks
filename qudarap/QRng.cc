#include <sstream>
#include <cstring>
#include "PLOG.hh"
#include "SPath.hh"
#include "QRng.hh"
#include "QU.hh"

#include "QUDA_CHECK.h"

const plog::Severity QRng::LEVEL = PLOG::EnvLevel("QRng", "INFO"); 
const QRng*          QRng::INSTANCE = nullptr ; 

const QRng* QRng::Get()
{
    return INSTANCE ;  
}

const char* QRng::DEFAULT_PATH = SPath::Resolve("$HOME/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin") ; 

QRng::QRng(const char* path_)
    :
    path(path_ ? strdup(path_) : DEFAULT_PATH),
    rngmax(0),
    d_rng_states(nullptr)
{
    INSTANCE = this ; 
    load_and_upload(); 
}

QRng::~QRng()
{
    QUDA_CHECK(cudaFree(d_rng_states)); 
}


/**
QRng::load
------------

Find that file_size is not a mutiple of item content. 
Presumably the 44 bytes of content get padded to 48 bytes
in the curandState which is typedef to curandStateXORWOW.

**/

void QRng::load_and_upload()
{
    FILE *fp = fopen(path,"rb");
    if(fp == NULL) {
        LOG(fatal) << " error opening file " << path ; 
        return ; 
    }   

    fseek(fp, 0L, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);

    long type_size = sizeof(curandState) ;  
    long content_size = 44 ; 

    rngmax = file_size/content_size ; 

    LOG(LEVEL) 
        << " path " << path 
        << " file_size " << file_size 
        << " type_size " << type_size 
        << " content_size " << content_size 
        << " rngmax " << rngmax
        ; 

    assert( file_size % content_size == 0 );  

    curandState* rng_states = (curandState*)malloc(sizeof(curandState)*rngmax);

    for(long i = 0 ; i < rngmax ; ++i )
    {   
        curandState& rng = rng_states[i] ;
        fread(&rng.d,                     sizeof(unsigned int),1,fp);   //  1
        fread(&rng.v,                     sizeof(unsigned int),5,fp);   //  5 
        fread(&rng.boxmuller_flag,        sizeof(int)         ,1,fp);   //  1 
        fread(&rng.boxmuller_flag_double, sizeof(int)         ,1,fp);   //  1
        fread(&rng.boxmuller_extra,       sizeof(float)       ,1,fp);   //  1
        fread(&rng.boxmuller_extra_double,sizeof(double)      ,1,fp);   //  2    11*4 = 44 
    }   
    fclose(fp);

    d_rng_states = QU::UploadArray<curandState>(rng_states, rngmax ) ;   

    free(rng_states); 
}


std::string QRng::desc() const
{
    std::stringstream ss ; 
    ss << "QRng"
       << " path " << path 
       << " rngmax " << rngmax 
       << " d_rng_states " << d_rng_states
       ;

    std::string s = ss.str(); 
    return s ; 
}

