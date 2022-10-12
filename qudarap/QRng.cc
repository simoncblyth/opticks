#include <sstream>
#include <cstring>
#include "SLOG.hh"
#include "SPath.hh"
#include "SCurandState.hh"
#include "QRng.hh"
#include "qrng.h"
#include "QU.hh"

#include "QUDA_CHECK.h"

const plog::Severity QRng::LEVEL = SLOG::EnvLevel("QRng", "DEBUG"); 
const QRng* QRng::INSTANCE = nullptr ; 
const QRng* QRng::Get(){ return INSTANCE ;  }


QRng::QRng(unsigned skipahead_event_offset)
    :
    path(SCurandState::Path()),        // null path will assert in Load
    rngmax(0),
    rng_states(Load(rngmax, path)),   // rngmax set based on file_size/item_size 
    qr(new qrng(skipahead_event_offset)),
    d_qr(nullptr)
{
    INSTANCE = this ; 
    upload(); 
    bool uploaded = d_qr != nullptr ; 
    LOG_IF(fatal, !uploaded) << " FAILED to upload curand states " ;  
    assert(uploaded); 
}


void QRng::cleanup()
{
    QUDA_CHECK(cudaFree(qr->rng_states)); 
}

QRng::~QRng()
{
}

const char* QRng::Load_FAIL_NOTES = R"(
QRng::Load_FAIL_NOTES
=======================

QRng::Load failed to load the curandState files. 
These files should to created during *opticks-full* installation 
by the bash function *opticks-prepare-installation* 
which runs *qudarap-prepare-installation*. 

Investigate by looking at the contents of the curandState directory, 
as shown below::

    epsilon:~ blyth$ ls -l  ~/.opticks/rngcache/RNG/
    total 892336
    -rw-r--r--  1 blyth  staff   44000000 Oct  6 19:43 QCurandState_1000000_0_0.bin
    -rw-r--r--  1 blyth  staff  132000000 Oct  6 19:53 QCurandState_3000000_0_0.bin
    epsilon:~ blyth$ 


)" ;

/**
QRng::Load
------------

rngmax is an output argument obtained from file_size/item_size 

**/

curandState* QRng::Load(long& rngmax, const char* path)  // static 
{
    bool null_path = path == nullptr ; 
    LOG_IF(fatal, null_path ) << " QRng::Load null path " ; 
    assert( !null_path );  

    FILE *fp = fopen(path,"rb");
    bool failed = fp == nullptr ; 
    LOG_IF(fatal, failed ) << " unabled to open file [" << path << "]" ; 
    LOG_IF(error, failed ) << Load_FAIL_NOTES  ; 
    assert(!failed); 


    fseek(fp, 0L, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);

    long type_size = sizeof(curandState) ;  
    long item_size = 44 ; 

    rngmax = file_size/item_size ; 


    LOG(LEVEL) 
        << " path " << path 
        << " file_size " << file_size 
        << " item_size " << item_size 
        << " type_size " << type_size 
        << " rngmax " << rngmax
        ; 

    assert( file_size % item_size == 0 );  

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

void QRng::Save( curandState* states, unsigned num_states, const char* path ) // static
{
    FILE *fp = fopen(path,"wb");
    LOG_IF(fatal, fp == nullptr) << " error opening file " << path ; 
    assert(fp); 

    for(unsigned i = 0 ; i < num_states ; ++i )
    {   
        curandState& rng = states[i] ;
        fwrite(&rng.d,                     sizeof(unsigned int),1,fp);
        fwrite(&rng.v,                     sizeof(unsigned int),5,fp);
        fwrite(&rng.boxmuller_flag,        sizeof(int)         ,1,fp);
        fwrite(&rng.boxmuller_flag_double, sizeof(int)         ,1,fp);
        fwrite(&rng.boxmuller_extra,       sizeof(float)       ,1,fp);
        fwrite(&rng.boxmuller_extra_double,sizeof(double)      ,1,fp);
    }   
    fclose(fp);
    return ; 
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
    T* d_uu = QU::device_alloc<T>(ni*nv, "QRng::generate:ni*nv");

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
    T* d_uu = QU::device_alloc<T>(ni*nv, "QRng::generate_2:ni*nv");

    QU::ConfigureLaunch(numBlocks, threadsPerBlock, ni, 1 );  

    QRng_generate_2<T>(numBlocks, threadsPerBlock, d_qr, event_idx, d_uu, ni, nv ); 

    QU::copy_device_to_host_and_free<T>( uu, d_uu, ni*nv );
}


template void QRng::generate_2<float>( float*,   unsigned, unsigned, unsigned ); 
template void QRng::generate_2<double>( double*, unsigned, unsigned, unsigned ); 





