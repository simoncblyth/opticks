#include <cassert>

#include "NP.hh"
#include "PLOG.hh"

#include "scuda.h"
#include "squad.h"
#include "srec.h"
#include "sseq.h"
#include "sphoton.h"
#include "sevent.h"

#include "QUDA_CHECK.h"
#include "QU.hh"
#include "curand_kernel.h"

#include "qsim.h"
#include "qbase.h"
#include "qprop.h"
#include "qrng.h"
#include "qdebug.h"
#include "qscint.h"
#include "qcerenkov.h"

#include "qmultifilm.h"


const plog::Severity QU::LEVEL = PLOG::EnvLevel("QU", "DEBUG") ; 


template <typename T> 
char QU::typecode()
{ 
    char c = '?' ; 
    switch(sizeof(T))
    {
        case 4: c = 'f' ; break ; 
        case 8: c = 'd' ; break ; 
    }
    return c ; 
}  



template <typename T>
std::string QU::rng_sequence_name(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ioffset ) // static 
{
    std::stringstream ss ; 
    ss << prefix
       << "_" << QU::typecode<T>()
       << "_ni" << ni 
       << "_nj" << nj 
       << "_nk" << nk 
       << "_ioffset" << std::setw(6) << std::setfill('0') << ioffset 
       << ".npy"
       ; 

    std::string name = ss.str(); 
    return name ; 
}

template std::string QU::rng_sequence_name<float>(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ioffset ) ; 
template std::string QU::rng_sequence_name<double>(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ioffset ) ; 



template <typename T>
std::string QU::rng_sequence_reldir(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size ) // static 
{
    std::stringstream ss ; 
    ss << prefix
       << "_" << QU::typecode<T>()
       << "_ni" << ni 
       << "_nj" << nj 
       << "_nk" << nk 
       << "_tranche" << ni_tranche_size 
       ; 

    std::string reldir = ss.str(); 
    return reldir ; 
}

template std::string QU::rng_sequence_reldir<float>(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size ) ; 
template std::string QU::rng_sequence_reldir<double>(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size ) ; 




/**
QU::UploadArray
----------------

Allocate on device and copy from host to device

**/

template <typename T>
T* QU::UploadArray(const T* array, unsigned num_items ) // static
{
    T* d_array = nullptr ; 
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_array ), num_items*sizeof(T) )); 
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_array ), array, sizeof(T)*num_items, cudaMemcpyHostToDevice )); 
    return d_array ; 
}

template float*         QU::UploadArray<float>(const float* array, unsigned num_items) ;
template unsigned*      QU::UploadArray<unsigned>(const unsigned* array, unsigned num_items) ;
template quad4*         QU::UploadArray<quad4>(const quad4* array, unsigned num_items) ;
template sphoton*       QU::UploadArray<sphoton>(const sphoton* array, unsigned num_items) ;
template quad2*         QU::UploadArray<quad2>(const quad2* array, unsigned num_items) ;
template curandState*   QU::UploadArray<curandState>(const curandState* array, unsigned num_items) ;
template qsim*          QU::UploadArray<qsim>(const qsim* array, unsigned num_items) ;
template qprop<float>*  QU::UploadArray<qprop<float>>(const qprop<float>* array, unsigned num_items) ;
template qprop<double>* QU::UploadArray<qprop<double>>(const qprop<double>* array, unsigned num_items) ;
template qmultifilm*    QU::UploadArray<qmultifilm>(const qmultifilm* array, unsigned num_items) ;
template qrng*          QU::UploadArray<qrng>(const qrng* array, unsigned num_items) ;
template qbnd*          QU::UploadArray<qbnd>(const qbnd* array, unsigned num_items) ;
template sevent*        QU::UploadArray<sevent>(const sevent* array, unsigned num_items) ;
template qdebug*        QU::UploadArray<qdebug>(const qdebug* array, unsigned num_items) ;
template qscint*        QU::UploadArray<qscint>(const qscint* array, unsigned num_items) ;
template qcerenkov*     QU::UploadArray<qcerenkov>(const qcerenkov* array, unsigned num_items) ;
template qbase*         QU::UploadArray<qbase>(const qbase* array, unsigned num_items) ;


/**
QU::DownloadArray  
-------------------

Allocate on host and copy from device to host 

**/

template <typename T>
T* QU::DownloadArray(const T* d_array, unsigned num_items ) // static
{
    T* array = new T[num_items] ;   
    QUDA_CHECK( cudaMemcpy( array, d_array, sizeof(T)*num_items, cudaMemcpyDeviceToHost )); 
    return array ; 
}


template  float*         QU::DownloadArray<float>(const float* d_array, unsigned num_items) ;
template  unsigned*      QU::DownloadArray<unsigned>(const unsigned* d_array, unsigned num_items) ;
template  quad4*         QU::DownloadArray<quad4>(const quad4* d_array, unsigned num_items) ;
template  quad2*         QU::DownloadArray<quad2>(const quad2* d_array, unsigned num_items) ;
template  curandState*   QU::DownloadArray<curandState>(const curandState* d_array, unsigned num_items) ;
template  qprop<float>*  QU::DownloadArray<qprop<float>>(const qprop<float>* d_array, unsigned num_items) ;
template  qprop<double>* QU::DownloadArray<qprop<double>>(const qprop<double>* d_array, unsigned num_items) ;


template <typename T>
void QU::Download(std::vector<T>& vec, const T* d_array, unsigned num_items)  // static
{
    vec.resize( num_items); 
    QUDA_CHECK( cudaMemcpy( static_cast<void*>( vec.data() ), d_array, num_items*sizeof(T), cudaMemcpyDeviceToHost)); 
}


template QUDARAP_API void QU::Download<float>(   std::vector<float>& vec,    const float* d_array,    unsigned num_items); 
template QUDARAP_API void QU::Download<unsigned>(std::vector<unsigned>& vec, const unsigned* d_array, unsigned num_items); 
template QUDARAP_API void QU::Download<int>(     std::vector<int>& vec,      const int* d_array,      unsigned num_items); 
template QUDARAP_API void QU::Download<uchar4>(  std::vector<uchar4>& vec,   const uchar4* d_array,   unsigned num_items); 
template QUDARAP_API void QU::Download<float4>(  std::vector<float4>& vec,   const float4* d_array,   unsigned num_items); 
template QUDARAP_API void QU::Download<quad4>(   std::vector<quad4>& vec,    const quad4* d_array,    unsigned num_items); 



template<typename T>
void QU::device_free_and_alloc(T** dd, unsigned num_items ) // dd: pointer-to-device-pointer
{
    size_t size = num_items*sizeof(T) ; 
    QUDA_CHECK( cudaFree( reinterpret_cast<void*>( *dd ) ) );
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( dd ), size )); 
    assert( *dd ); 
}


template QUDARAP_API void  QU::device_free_and_alloc<float>(float** dd, unsigned num_items) ;
template QUDARAP_API void  QU::device_free_and_alloc<double>(double** dd, unsigned num_items) ;
template QUDARAP_API void  QU::device_free_and_alloc<unsigned>(unsigned** dd, unsigned num_items) ;
template QUDARAP_API void  QU::device_free_and_alloc<int>(int** dd, unsigned num_items) ;
template QUDARAP_API void  QU::device_free_and_alloc<quad>(quad** dd, unsigned num_items) ;
template QUDARAP_API void  QU::device_free_and_alloc<uchar4>(uchar4** dd, unsigned num_items) ;
template QUDARAP_API void  QU::device_free_and_alloc<float4>(float4** dd, unsigned num_items) ;
template QUDARAP_API void  QU::device_free_and_alloc<quad4>(quad4** dd, unsigned num_items) ;



template<typename T>
T* QU::device_alloc( unsigned num_items )
{
    size_t size = num_items*sizeof(T) ; 
    T* d ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d ), size )); 

    return d ; 
}

template QUDARAP_API float*     QU::device_alloc<float>(unsigned num_items) ;
template QUDARAP_API double*    QU::device_alloc<double>(unsigned num_items) ;
template QUDARAP_API unsigned*  QU::device_alloc<unsigned>(unsigned num_items) ;
template QUDARAP_API int*       QU::device_alloc<int>(unsigned num_items) ;
template QUDARAP_API uchar4*    QU::device_alloc<uchar4>(unsigned num_items) ;
template QUDARAP_API float4*    QU::device_alloc<float4>(unsigned num_items) ;
template QUDARAP_API quad*      QU::device_alloc<quad>(unsigned num_items) ;
template QUDARAP_API quad2*     QU::device_alloc<quad2>(unsigned num_items) ;
template QUDARAP_API quad4*     QU::device_alloc<quad4>(unsigned num_items) ;
template QUDARAP_API quad6*     QU::device_alloc<quad6>(unsigned num_items) ;
template QUDARAP_API sevent*    QU::device_alloc<sevent>(unsigned num_items) ;
template QUDARAP_API qdebug*    QU::device_alloc<qdebug>(unsigned num_items) ;
template QUDARAP_API qstate*    QU::device_alloc<qstate>(unsigned num_items) ;
template QUDARAP_API srec*      QU::device_alloc<srec>(unsigned num_items) ;
template QUDARAP_API sseq*      QU::device_alloc<sseq>(unsigned num_items) ;
template QUDARAP_API sphoton*   QU::device_alloc<sphoton>(unsigned num_items) ;


template<typename T>
T* QU::device_alloc_zero(unsigned num_items )
{
    size_t size = num_items*sizeof(T) ; 
    T* d ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d ), size )); 

    int value = 0 ; 
    QUDA_CHECK( cudaMemset(d, value, size )); 

    return d ; 
}

template QUDARAP_API sphoton*   QU::device_alloc_zero<sphoton>(unsigned num_items) ;
template QUDARAP_API srec*      QU::device_alloc_zero<srec>(unsigned num_items) ;
template QUDARAP_API sseq*      QU::device_alloc_zero<sseq>(unsigned num_items) ;




template<typename T>
void QU::device_memset( T* d, int value, unsigned num_items )
{
    size_t size = num_items*sizeof(T) ; 
    QUDA_CHECK( cudaMemset(d, value, size )); 
}

template void     QU::device_memset<int>(int*, int, unsigned ) ;
template void     QU::device_memset<quad4>(quad4*, int, unsigned ) ;
template void     QU::device_memset<quad6>(quad6*, int, unsigned ) ;
template void     QU::device_memset<sphoton>(sphoton*, int, unsigned ) ;










template<typename T>
void QU::device_free( T* d)
{
    QUDA_CHECK( cudaFree(d) ); 
}

template void   QU::device_free<float>(float*) ;
template void   QU::device_free<double>(double*) ;
template void   QU::device_free<unsigned>(unsigned*) ;
template void   QU::device_free<quad2>(quad2*) ;
template void   QU::device_free<quad4>(quad4*) ;
template void   QU::device_free<sphoton>(sphoton*) ;


template<typename T>
void QU::copy_device_to_host( T* h, T* d,  unsigned num_items)
{
    size_t size = num_items*sizeof(T) ; 
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost )); 
}


template void QU::copy_device_to_host<int>(  int* h, int* d,  unsigned num_items);
template void QU::copy_device_to_host<float>(  float* h, float* d,  unsigned num_items);
template void QU::copy_device_to_host<double>( double* h, double* d,  unsigned num_items);
template void QU::copy_device_to_host<quad>( quad* h, quad* d,  unsigned num_items);
template void QU::copy_device_to_host<quad2>( quad2* h, quad2* d,  unsigned num_items);
template void QU::copy_device_to_host<quad4>( quad4* h, quad4* d,  unsigned num_items);
template void QU::copy_device_to_host<sphoton>( sphoton* h, sphoton* d,  unsigned num_items);
template void QU::copy_device_to_host<quad6>( quad6* h, quad6* d,  unsigned num_items);
template void QU::copy_device_to_host<qstate>( qstate* h, qstate* d,  unsigned num_items);
template void QU::copy_device_to_host<srec>( srec* h, srec* d,  unsigned num_items);
template void QU::copy_device_to_host<sseq>( sseq* h, sseq* d,  unsigned num_items);


/**
QU::copy_device_to_host_and_free
----------------------------------

* Summary: when you get cudaMemcpy copyback errors look for infinite loops in kernels
* Find the problem by doing things like adding loop limiters

When a kernel misbehaves, such as going into an infinite loop for example, the
connection to the GPU might timeout. Subsequent attempts to copyback arrays that 
should have been written by the kernel would then fail during the cudaMemcpy 
presumably because the CUDA context is lost as a result of the timeout making 
all the device pointers invalid. The copyback is the usual thing to fail because 
it is the normally the first thing to use the stale pointers after the kernel launch. 

**/

template<typename T>
void QU::copy_device_to_host_and_free( T* h, T* d,  unsigned num_items)
{
    size_t size = num_items*sizeof(T) ; 
    LOG(info) << "copy " << num_items << " sizeof(T) " << sizeof(T) ;  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d) ); 
}


template void QU::copy_device_to_host_and_free<float>(  float* h, float* d,  unsigned num_items);
template void QU::copy_device_to_host_and_free<double>( double* h, double* d,  unsigned num_items);
template void QU::copy_device_to_host_and_free<quad>( quad* h, quad* d,  unsigned num_items);
template void QU::copy_device_to_host_and_free<quad2>( quad2* h, quad2* d,  unsigned num_items);
template void QU::copy_device_to_host_and_free<quad4>( quad4* h, quad4* d,  unsigned num_items);
template void QU::copy_device_to_host_and_free<sphoton>( sphoton* h, sphoton* d,  unsigned num_items);
template void QU::copy_device_to_host_and_free<quad6>( quad6* h, quad6* d,  unsigned num_items);
template void QU::copy_device_to_host_and_free<qstate>( qstate* h, qstate* d,  unsigned num_items);












template<typename T>
void QU::copy_host_to_device( T* d, const T* h, unsigned num_items)
{
    size_t size = num_items*sizeof(T) ; 
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d ), h , size, cudaMemcpyHostToDevice )); 
}

template void QU::copy_host_to_device<float>(    float* d,   const float* h, unsigned num_items);
template void QU::copy_host_to_device<double>(   double* d,  const double* h, unsigned num_items);
template void QU::copy_host_to_device<unsigned>( unsigned* d, const unsigned* h, unsigned num_items);
template void QU::copy_host_to_device<sevent>(   sevent* d,   const sevent* h, unsigned num_items);
template void QU::copy_host_to_device<quad4>(    quad4* d,    const quad4* h, unsigned num_items);
template void QU::copy_host_to_device<sphoton>(  sphoton* d,  const sphoton* h, unsigned num_items);
template void QU::copy_host_to_device<quad6>(    quad6* d,    const quad6* h, unsigned num_items);
template void QU::copy_host_to_device<quad2>(    quad2* d,    const quad2* h, unsigned num_items);

/**
QU::NumItems
---------------

Apply heuristics to determine the number of intended GPU buffer items
using the size of the template type and the shape of the NP array. 

**/

template <typename T>
unsigned QU::NumItems( const NP* a )
{
    unsigned num_items = 0 ; 

    if( sizeof(T) == sizeof(float)*6*4 )   // looks like quad6
    {
        if(a->shape.size() == 3 )
        {
            assert( a->has_shape( -1, 6, 4) ); 
            num_items = a->shape[0] ; 
        }
    }
    else if( sizeof(T) == sizeof(float)*4*4 )   // looks like quad4 
    {
        if(a->shape.size() == 3 )
        {
            assert( a->has_shape( -1, 4, 4) ); 
            num_items = a->shape[0] ; 
        }
        else if(a->shape.size() == 4 )
        {
            assert( a->shape[2] == 2 && a->shape[3] == 4 ); 
            num_items = a->shape[0]*a->shape[1] ; 
        }
    }  
    else if( sizeof(T) == sizeof(float)*4*2 ) // looks like quad2
    {
        if(a->shape.size() == 3 )
        {
            assert( a->has_shape( -1, 2, 4) ); 
            num_items = a->shape[0] ; 
        }
        else if(a->shape.size() == 4 )
        {
            assert( a->shape[2] == 2 && a->shape[3] == 4 ); 
            num_items = a->shape[0]*a->shape[1] ; 
        }
    }  
    return num_items ; 
}

template unsigned QU::NumItems<quad2>(const NP* );
template unsigned QU::NumItems<quad4>(const NP* );
template unsigned QU::NumItems<quad6>(const NP* );


/**
QU::copy_host_to_device
------------------------

HMM: encapsulating determination of num_items is less useful than 
would initially expect because will always need to know 
and record the num_items in a shared GPU/CPU location like sevent.
And also will often need to allocate the buffer first too. 

Suggesting should generally use this via QEvent. 

**/

template <typename T>
unsigned QU::copy_host_to_device( T* d, const NP* a)
{
    unsigned num_items = NumItems<T>(a); 
    if( num_items == 0 )
    {
        LOG(fatal) << " failed to devine num_items for array " << a->sstr() << " with template type where sizeof(T) " << sizeof(T) ; 
    }

    if( num_items > 0 )
    {
        copy_host_to_device( d, (T*)a->bytes(), num_items ); 
    }
    return num_items ; 
}

template unsigned QU::copy_host_to_device<quad2>( quad2* , const NP* );   
template unsigned QU::copy_host_to_device<quad4>( quad4* , const NP* );   
template unsigned QU::copy_host_to_device<quad6>( quad6* , const NP* );   





/**
QU::ConfigureLaunch
---------------------




**/

void QU::ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height ) // static
{
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 

    // hmm this looks to not handle height other than 1 
}

void QU::ConfigureLaunch1D( dim3& numBlocks, dim3& threadsPerBlock, unsigned num, unsigned threads_per_block ) // static
{
    threadsPerBlock.x = threads_per_block ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (num + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = 1 ; 
    numBlocks.z = 1 ; 
}



void QU::ConfigureLaunch2D( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height ) // static
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 16 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}


void QU::ConfigureLaunch16( dim3& numBlocks, dim3& threadsPerBlock ) // static
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 

    numBlocks.x = 1 ; 
    numBlocks.y = 1 ; 
    numBlocks.z = 1 ; 
}


std::string QU::Desc(const dim3& d, int w) // static 
{
    std::stringstream ss ; 
    ss << "( " 
        << std::setw(w) << d.x 
        << " " 
        << std::setw(w) << d.y 
        << " " 
        << std::setw(w) << d.z 
        << ")"
        ;
    std::string s = ss.str(); 
    return s ; 
}

std::string QU::DescLaunch( const dim3& numBlocks, const dim3& threadsPerBlock ) // static
{
    std::stringstream ss ; 
    ss 
        << " numBlocks " << Desc(numBlocks,4) 
        << " threadsPerBlock " << Desc(threadsPerBlock, 4)
        ;
    std::string s = ss.str(); 
    return s ; 
}

