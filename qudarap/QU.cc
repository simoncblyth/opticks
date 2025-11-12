#include <cassert>

#include "NP.hh"
#include "SLOG.hh"

#include "spath.h"
#include "sdirectory.h"
#include "scuda.h"
#include "squad.h"
#include "ssys.h"

#ifndef PRODUCTION
#include "srec.h"
#include "sseq.h"
#endif

#include "sphoton.h"
#include "sphotonlite.h"

#include "sevent.h"
#include "salloc.h"
#include "SEventConfig.hh"

#include "QUDA_CHECK.h"
#include "QU.hh"

#include "curand_kernel.h"
#include "qrng.h"
#include "qsim.h"

#include "qbase.h"
#include "qprop.h"
#include "qpmt.h"
#include "qdebug.h"
#include "qscint.h"
#include "qcerenkov.h"
#include "qcurandwrap.h"
#include "scurandref.h"
#include "qmultifilm.h"


const plog::Severity QU::LEVEL = SLOG::EnvLevel("QU", "DEBUG") ;
bool QU::MEMCHECK = ssys::getenvbool(_MEMCHECK);

salloc* QU::alloc = nullptr ;


void QU::alloc_add(const char* label, uint64_t num_items, uint64_t sizeof_item ) // static
{
   if(!alloc) alloc = SEventConfig::ALLOC ;
   if(alloc ) alloc->add(label, num_items, sizeof_item );
}


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

template char QU::typecode<float>() ;
template char QU::typecode<double>() ;


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
T* QU::UploadArray(const T* array, unsigned num_items, const char* label ) // static
{
    size_t size = num_items*sizeof(T) ;

    LOG(LEVEL)
       << " num_items " << num_items
       << " size " << size
       << " label " << ( label ? label : "-" )
       ;

    LOG_IF(info, MEMCHECK)
       << " num_items " << num_items
       << " size " << size
       << " label " << ( label ? label : "-" )
       ;


    alloc_add( label, num_items, sizeof(T) ) ;

    T* d_array = nullptr ;
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_array ), size ));
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_array ), array, size, cudaMemcpyHostToDevice ));
    return d_array ;
}


// IF NEED THESE FROM REMOVE PKG WILL NEED TO QUDARAP_API
template float*         QU::UploadArray<float>(const float* array, unsigned num_items, const char* label ) ;
template double*        QU::UploadArray<double>(const double* array, unsigned num_items, const char* label) ;
template unsigned*      QU::UploadArray<unsigned>(const unsigned* array, unsigned num_items, const char* label) ;
template int*           QU::UploadArray<int>(const int* array, unsigned num_items, const char* label) ;
template quad4*         QU::UploadArray<quad4>(const quad4* array, unsigned num_items, const char* label) ;
template sphoton*       QU::UploadArray<sphoton>(const sphoton* array, unsigned num_items, const char* label) ;
template quad2*         QU::UploadArray<quad2>(const quad2* array, unsigned num_items, const char* label) ;
template XORWOW*        QU::UploadArray<XORWOW>(const XORWOW* array, unsigned num_items, const char* label) ;
template Philox*        QU::UploadArray<Philox>(const Philox* array, unsigned num_items, const char* label) ;
template qcurandwrap<XORWOW>*   QU::UploadArray<qcurandwrap<XORWOW>>(const qcurandwrap<XORWOW>* array, unsigned num_items, const char* label) ;
template scurandref<XORWOW>*    QU::UploadArray<scurandref<XORWOW>>(const scurandref<XORWOW>* array, unsigned num_items, const char* label) ;
template qsim*          QU::UploadArray<qsim>(const qsim* array, unsigned num_items, const char* label) ;
template qprop<float>*  QU::UploadArray<qprop<float>>(const qprop<float>* array, unsigned num_items, const char* label) ;
template qprop<double>* QU::UploadArray<qprop<double>>(const qprop<double>* array, unsigned num_items, const char* label) ;
template qpmt<float>*   QU::UploadArray<qpmt<float>>(const qpmt<float>* array, unsigned num_items, const char* label) ;
template qpmt<double>*  QU::UploadArray<qpmt<double>>(const qpmt<double>* array, unsigned num_items, const char* label) ;
template qmultifilm*    QU::UploadArray<qmultifilm>(const qmultifilm* array, unsigned num_items, const char* label) ;
template qrng<RNG>*     QU::UploadArray<qrng<RNG>>(const qrng<RNG>* array, unsigned num_items, const char* label) ;
template qbnd*          QU::UploadArray<qbnd>(const qbnd* array, unsigned num_items, const char* label) ;
template sevent*        QU::UploadArray<sevent>(const sevent* array, unsigned num_items, const char* label) ;
template qdebug*        QU::UploadArray<qdebug>(const qdebug* array, unsigned num_items, const char* label) ;
template qscint*        QU::UploadArray<qscint>(const qscint* array, unsigned num_items, const char* label) ;
template qcerenkov*     QU::UploadArray<qcerenkov>(const qcerenkov* array, unsigned num_items, const char* label) ;
template qbase*         QU::UploadArray<qbase>(const qbase* array, unsigned num_items, const char* label) ;



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
template  int*           QU::DownloadArray<int>(const int* d_array, unsigned num_items) ;
template  quad4*         QU::DownloadArray<quad4>(const quad4* d_array, unsigned num_items) ;
template  quad2*         QU::DownloadArray<quad2>(const quad2* d_array, unsigned num_items) ;
template  XORWOW*        QU::DownloadArray<XORWOW>(const XORWOW* d_array, unsigned num_items) ;
template  Philox*        QU::DownloadArray<Philox>(const Philox* d_array, unsigned num_items) ;
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
    LOG_IF(info, MEMCHECK) << " size " << size << " num_items " << num_items ;

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

const char* QU::_cudaMalloc_OOM_NOTES = R"( ;
QU::_cudaMalloc_OOM_NOTES
==========================

When running with debug arrays, such as the record array, enabled
it is necessary to set max_slot to something reasonable, otherwise with the
default max_slot of zero, it gets set to a high value (eg M197 with 24GB)
appropriate for production running with the available VRAM.

One million is typically reasonable for debugging::

   export OPTICKS_MAX_SLOT=M1

)" ;




void QU::_cudaMalloc( void** p2p, size_t size, const char* label )
{
    cudaError_t err = cudaMalloc(p2p, size ) ;
    if( err != cudaSuccess )
    {
        const char* out = spath::Resolve("$DefaultOutputDir") ;
        salloc* estimate = SEventConfig::AllocEstimate();

        std::stringstream ss;
        ss << "CUDA call (" << label << " ) failed with error: '"
           << cudaGetErrorString( err )
           << "' (" __FILE__ << ":" << __LINE__ << ")"
           << "\n\n"
           << "[SEventConfig::DescEventMode (use of DebugHeavy/DebugLite EventMode with high stats is typical cause of OOM errors)\n"
           << SEventConfig::DescEventMode()
           << "]SEventConfig::DescEventMode (use of DebugHeavy/DebugLite EventMode with high stats is typical cause of OOM errors)\n"
           << "\n\n"
           << "[alloc.desc\n"
           << ( alloc ? alloc->desc() : "no-alloc" )
           << "]alloc.desc\n"
           << "\n"
           << "[NOTES\n"
           << _cudaMalloc_OOM_NOTES
           << "]NOTES\n"
           << "\n\n"
           << "[SEventConfig::AllocEstimate\n"
           << ( estimate ? estimate->desc() : "no-estimate" )
           << "]SEventConfig::AllocEstimate\n"
           << "save salloc record to [" << out << "]\n" ;
           ;

        std::string msg = ss.str();
        LOG(error) << msg ;

        sdirectory::MakeDirs(out,0);
        alloc->save(out) ;

        throw QUDA_Exception( msg.c_str() );
    }
}


template<typename T>
T* QU::device_alloc( unsigned num_items, const char* label )
{
    size_t size = num_items*sizeof(T) ;

    LOG(LEVEL)
        << " num_items " << std::setw(10) << num_items
        << " size " << std::setw(10) << size
        << " label " << std::setw(15) << label
        ;

    LOG_IF(info, MEMCHECK)
        << " num_items " << std::setw(10) << num_items
        << " size " << std::setw(10) << size
        << " label " << std::setw(15) << label
        ;


    alloc_add( label, num_items, sizeof(T) ) ;

    T* d ;
    _cudaMalloc( reinterpret_cast<void**>( &d ), size, label );

    return d ;
}

template QUDARAP_API float*     QU::device_alloc<float>(unsigned num_items, const char* label) ;
template QUDARAP_API double*    QU::device_alloc<double>(unsigned num_items, const char* label) ;
template QUDARAP_API unsigned*  QU::device_alloc<unsigned>(unsigned num_items, const char* label) ;
template QUDARAP_API int*       QU::device_alloc<int>(unsigned num_items, const char* label) ;
template QUDARAP_API uchar4*    QU::device_alloc<uchar4>(unsigned num_items, const char* label) ;
template QUDARAP_API float4*    QU::device_alloc<float4>(unsigned num_items, const char* label) ;
template QUDARAP_API quad*      QU::device_alloc<quad>(unsigned num_items, const char* label) ;
template QUDARAP_API quad2*     QU::device_alloc<quad2>(unsigned num_items, const char* label) ;
template QUDARAP_API quad4*     QU::device_alloc<quad4>(unsigned num_items, const char* label) ;
template QUDARAP_API quad6*     QU::device_alloc<quad6>(unsigned num_items, const char* label) ;
template QUDARAP_API sevent*    QU::device_alloc<sevent>(unsigned num_items, const char* label) ;
template QUDARAP_API qdebug*    QU::device_alloc<qdebug>(unsigned num_items, const char* label) ;
template QUDARAP_API sstate*    QU::device_alloc<sstate>(unsigned num_items, const char* label) ;
template QUDARAP_API XORWOW*    QU::device_alloc<XORWOW>(unsigned num_items, const char* label) ;
template QUDARAP_API Philox*    QU::device_alloc<Philox>(unsigned num_items, const char* label) ;

#ifndef PRODUCTION
template QUDARAP_API srec*      QU::device_alloc<srec>(unsigned num_items, const char* label) ;
template QUDARAP_API sseq*      QU::device_alloc<sseq>(unsigned num_items, const char* label) ;
#endif

template QUDARAP_API sphoton*   QU::device_alloc<sphoton>(unsigned num_items, const char* label) ;
template QUDARAP_API sphotonlite*   QU::device_alloc<sphotonlite>(unsigned num_items, const char* label) ;


template<typename T>
T* QU::device_alloc_zero(unsigned num_items, const char* label)
{
    size_t size = num_items*sizeof(T) ;

    LOG(LEVEL)
        << " num_items " << std::setw(10) << num_items
        << " sizeof(T) " << std::setw(10) << sizeof(T)
        << " size " << std::setw(10) << size
        << " label " << std::setw(15) << label
        ;

    LOG_IF(info, MEMCHECK)
        << " num_items " << std::setw(10) << num_items
        << " sizeof(T) " << std::setw(10) << sizeof(T)
        << " size " << std::setw(10) << size
        << " label " << std::setw(15) << label
        ;


    alloc_add( label, num_items, sizeof(T) ) ;

    T* d ;
    _cudaMalloc( reinterpret_cast<void**>( &d ), size, label );

    int value = 0 ;
    QUDA_CHECK( cudaMemset(d, value, size ));

    return d ;
}

template QUDARAP_API sphoton*   QU::device_alloc_zero<sphoton>(unsigned num_items, const char* label) ;
template QUDARAP_API sphotonlite*   QU::device_alloc_zero<sphotonlite>(unsigned num_items, const char* label) ;
template QUDARAP_API quad2*     QU::device_alloc_zero<quad2>(  unsigned num_items, const char* label) ;
template QUDARAP_API XORWOW*    QU::device_alloc_zero<XORWOW>(  unsigned num_items, const char* label) ;
template QUDARAP_API Philox*    QU::device_alloc_zero<Philox>(  unsigned num_items, const char* label) ;

#ifndef PRODUCTION
template QUDARAP_API srec*      QU::device_alloc_zero<srec>(   unsigned num_items, const char* label) ;
template QUDARAP_API sseq*      QU::device_alloc_zero<sseq>(   unsigned num_items, const char* label) ;
template QUDARAP_API stag*      QU::device_alloc_zero<stag>(   unsigned num_items, const char* label) ;
template QUDARAP_API sflat*     QU::device_alloc_zero<sflat>(  unsigned num_items, const char* label) ;
#endif




template<typename T>
void QU::device_memset( T* d, int value, unsigned num_items )
{
    size_t size = num_items*sizeof(T) ;

    LOG_IF(info, MEMCHECK)
        << " num_items " << std::setw(10) << num_items
        << " sizeof(T) " << std::setw(10) << sizeof(T)
        << " size " << std::setw(10) << size
        ;

    QUDA_CHECK( cudaMemset(d, value, size ));
}

template QUDARAP_API void     QU::device_memset<int>(int*, int, unsigned ) ;
template QUDARAP_API void     QU::device_memset<quad4>(quad4*, int, unsigned ) ;
template QUDARAP_API void     QU::device_memset<quad6>(quad6*, int, unsigned ) ;
template QUDARAP_API void     QU::device_memset<sphoton>(sphoton*, int, unsigned ) ;
template QUDARAP_API void     QU::device_memset<sphotonlite>(sphotonlite*, int, unsigned ) ;










template<typename T>
void QU::device_free( T* d)
{
    LOG_IF(info, MEMCHECK) ;
    // HMM: could use salloc to find the label ?

    QUDA_CHECK( cudaFree(d) );
}

template QUDARAP_API void   QU::device_free<float>(float*) ;
template QUDARAP_API void   QU::device_free<double>(double*) ;
template QUDARAP_API void   QU::device_free<unsigned>(unsigned*) ;
template QUDARAP_API void   QU::device_free<quad2>(quad2*) ;
template QUDARAP_API void   QU::device_free<quad4>(quad4*) ;
template QUDARAP_API void   QU::device_free<sphoton>(sphoton*) ;
template QUDARAP_API void   QU::device_free<sphotonlite>(sphotonlite*) ;
template QUDARAP_API void   QU::device_free<uchar4>(uchar4*) ;
template QUDARAP_API void   QU::device_free<XORWOW>(XORWOW*) ;
template QUDARAP_API void   QU::device_free<Philox>(Philox*) ;


template<typename T>
int QU::copy_device_to_host( T* h, T* d,  unsigned num_items)
{
    if( d == nullptr ) std::cerr
        << "QU::copy_device_to_host"
        << " ERROR : device pointer is null "
        << std::endl
        ;

    if( d == nullptr ) return 1 ;

    size_t size = num_items*sizeof(T) ;
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost ));

    return 0 ;
}


template int QU::copy_device_to_host<int>(  int* h, int* d,  unsigned num_items);
template int QU::copy_device_to_host<float>(  float* h, float* d,  unsigned num_items);
template int QU::copy_device_to_host<double>( double* h, double* d,  unsigned num_items);
template int QU::copy_device_to_host<quad>( quad* h, quad* d,  unsigned num_items);
template int QU::copy_device_to_host<quad2>( quad2* h, quad2* d,  unsigned num_items);
template int QU::copy_device_to_host<quad4>( quad4* h, quad4* d,  unsigned num_items);
template int QU::copy_device_to_host<sphoton>( sphoton* h, sphoton* d,  unsigned num_items);
template int QU::copy_device_to_host<sphotonlite>( sphotonlite* h, sphotonlite* d,  unsigned num_items);
template int QU::copy_device_to_host<quad6>( quad6* h, quad6* d,  unsigned num_items);
template int QU::copy_device_to_host<sstate>( sstate* h, sstate* d,  unsigned num_items);
template int QU::copy_device_to_host<XORWOW>( XORWOW* h, XORWOW* d,  unsigned num_items);
template int QU::copy_device_to_host<Philox>( Philox* h, Philox* d,  unsigned num_items);
#ifndef PRODUCTION
template int QU::copy_device_to_host<srec>( srec* h, srec* d,  unsigned num_items);
template int QU::copy_device_to_host<sseq>( sseq* h, sseq* d,  unsigned num_items);
template int QU::copy_device_to_host<stag>( stag* h, stag* d,  unsigned num_items);
template int QU::copy_device_to_host<sflat>( sflat* h, sflat* d,  unsigned num_items);
#endif


/**
QU::copy_device_to_host_and_free
----------------------------------

* Summary: when you get cudaMemcpy copyback errors look for infinite loops in kernels
* Find the problem by doing things like adding loop limiters


Normally the problem is not related to the copying but rather some issue
with the kernel being called. So start by doing "binary" search
simplifying the kernel to find where the issue is.

When a kernel misbehaves, such as going into an infinite loop for example, the
connection to the GPU will typically timeout. Subsequent attempts to copyback arrays that
should have been written by the kernel would then fail during the cudaMemcpy
presumably because the CUDA context is lost as a result of the timeout making
all the device pointers invalid. The copyback is the usual thing to fail because
it is the normally the first thing to use the stale pointers after the kernel launch.


Debug tip 0
~~~~~~~~~~~~~

Simply add "return 0" to call with issue, and
progressivley move that forwards to find where
the issue is.


Debug tip 1 : check kernel inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of doing whatever computation in the kernel,
populate the output array with the inputs.
This checks both having expected inputs at the kernel
and the copy out machinery.

Debug tip 2 : check intermediate kernel results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intead of doing the full kernel calculation, check the
first half of the calculation by copying intermediate
results into the output array.


**/

template<typename T>
void QU::copy_device_to_host_and_free( T* h, T* d,  unsigned num_items, const char* label)
{
    size_t size = num_items*sizeof(T) ;
    LOG(LEVEL)
        << "copy " << num_items
        << " sizeof(T) " << sizeof(T)
        << " label " << ( label ? label : "-" )
        ;

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost ));
    QUDA_CHECK( cudaFree(d) );
}


template void QU::copy_device_to_host_and_free<float>(  float* h, float* d,  unsigned num_items, const char* label );
template void QU::copy_device_to_host_and_free<double>( double* h, double* d,  unsigned num_items, const char* label);
template void QU::copy_device_to_host_and_free<quad>( quad* h, quad* d,  unsigned num_items, const char* label);
template void QU::copy_device_to_host_and_free<quad2>( quad2* h, quad2* d,  unsigned num_items, const char* label);
template void QU::copy_device_to_host_and_free<quad4>( quad4* h, quad4* d,  unsigned num_items, const char* label);
template void QU::copy_device_to_host_and_free<sphoton>( sphoton* h, sphoton* d,  unsigned num_items, const char* label);
template void QU::copy_device_to_host_and_free<sphotonlite>( sphotonlite* h, sphotonlite* d,  unsigned num_items, const char* label);
template void QU::copy_device_to_host_and_free<quad6>( quad6* h, quad6* d,  unsigned num_items, const char* label);
template void QU::copy_device_to_host_and_free<sstate>( sstate* h, sstate* d,  unsigned num_items, const char* label);












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
template void QU::copy_host_to_device<sphotonlite>(  sphotonlite* d,  const sphotonlite* h, unsigned num_items);
template void QU::copy_host_to_device<quad6>(    quad6* d,    const quad6* h, unsigned num_items);
template void QU::copy_host_to_device<quad2>(    quad2* d,    const quad2* h, unsigned num_items);
template void QU::copy_host_to_device<XORWOW>(   XORWOW* d,   const XORWOW* h,   unsigned num_items);
template void QU::copy_host_to_device<Philox>(   Philox* d,   const Philox* h,   unsigned num_items);

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

Suggesting should generally use this via QEvt.

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

