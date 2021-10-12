#include <sstream>
#include <cstring>
#include "PLOG.hh"
#include "SPath.hh"
#include "QRng.hh"
#include "qrng.h"
#include "QU.hh"

#include "QUDA_CHECK.h"

const plog::Severity QRng::LEVEL = PLOG::EnvLevel("QRng", "INFO"); 
const QRng* QRng::INSTANCE = nullptr ; 
const QRng* QRng::Get(){ return INSTANCE ;  }

const char* QRng::DEFAULT_PATH = SPath::Resolve("$HOME/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin", false) ; 
//const char* QRng::DEFAULT_PATH = SPath::Resolve("$HOME/.opticks/rngcache/RNG/cuRANDWrapper_3000000_0_0.bin", false) ; 

QRng::QRng(const char* path_, unsigned skipahead_event_offset)
    :
    path(path_ ? strdup(path_) : DEFAULT_PATH),
    rngmax(0),
    rng_states(Load(rngmax, path)),
    qr(new qrng(skipahead_event_offset)),
    d_qr(nullptr)
{
    INSTANCE = this ; 
    upload(); 
}

QRng::~QRng()
{
    QUDA_CHECK(cudaFree(qr->rng_states)); 
}


/**
QRng::Load
------------

Find that file_size is not a mutiple of item content. 
Presumably the 44 bytes of content get padded to 48 bytes
in the curandState which is typedef to curandStateXORWOW.

**/

curandState* QRng::Load(long& rngmax, const char* path)  // static 
{
    FILE *fp = fopen(path,"rb");
    if(fp == NULL) {
        LOG(fatal) << " error opening file " << path ; 
        return nullptr ; 
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

    return rng_states ; 
}



void QRng::upload()
{
    qr->rng_states = QU::UploadArray<curandState>(rng_states, rngmax ) ;   

    free(rng_states); 
    rng_states = nullptr ; 

    d_qr = QU::UploadArray<qrng>(qr, 1 ); 
}


std::string QRng::desc() const
{
    std::stringstream ss ; 
    ss << "QRng"
       << " path " << path 
       << " rngmax " << rngmax 
       << " qr " << qr
       << " d_qr " << d_qr
       ;

    std::string s = ss.str(); 
    return s ; 
}



template <typename T>
extern void QRng_generate(
    dim3, 
    dim3, 
    T*, 
    unsigned,
    unsigned,
    curandState*,
    unsigned long long 
);


template<typename T>
void QRng::generate( T* uu, unsigned ni, unsigned nv, unsigned long long skipahead_ )
{
    T* d_uu = QU::device_alloc<T>(ni*nv);

    QU::ConfigureLaunch(numBlocks, threadsPerBlock, ni, 1 );  

    QRng_generate<T>(numBlocks, threadsPerBlock, d_uu, ni, nv, qr->rng_states, skipahead_ ); 

    QU::copy_device_to_host_and_free<T>( uu, d_uu, ni*nv );
}


template void QRng::generate<float>( float*,   unsigned, unsigned, unsigned long long ); 
template void QRng::generate<double>( double*, unsigned, unsigned, unsigned long long ); 








template <typename T>
extern void QRng_generate_2(dim3, dim3, qrng*, unsigned, T*, unsigned, unsigned );


template<typename T>
void QRng::generate_2( T* uu, unsigned ni, unsigned nv, unsigned event_idx )
{
    T* d_uu = QU::device_alloc<T>(ni*nv);

    QU::ConfigureLaunch(numBlocks, threadsPerBlock, ni, 1 );  

    QRng_generate_2<T>(numBlocks, threadsPerBlock, d_qr, event_idx, d_uu, ni, nv ); 

    QU::copy_device_to_host_and_free<T>( uu, d_uu, ni*nv );
}


template void QRng::generate_2<float>( float*,   unsigned, unsigned, unsigned ); 
template void QRng::generate_2<double>( double*, unsigned, unsigned, unsigned ); 





