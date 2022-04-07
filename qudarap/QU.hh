#pragma once
#include "QUDARAP_API_EXPORT.hh"
#include <vector>

struct QUDARAP_API QU
{
    template <typename T>
    static char typecode() ; 

    template <typename T>
    static std::string rng_sequence_reldir(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size );

    template <typename T>
    static std::string rng_sequence_name(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ioffset );

    template <typename T>
    static T* UploadArray(const T* array, unsigned num_items ) ;    

    template <typename T>
    static T* DownloadArray(const T* array, unsigned num_items ) ;    

    template <typename T>
    static void Download(std::vector<T>& vec, const T* d_array, unsigned num_items) ;    

    template <typename T>
    static T*   device_alloc( unsigned num_items ) ; 

    template <typename T>
    static void device_memset( T* d, int value, unsigned num_items ); 

    template <typename T>
    static void device_free( T* d ) ; 

    template <typename T>
    static void device_free_and_alloc(T** dd, unsigned num_items );  // dd : pointer-to-device-pointer

    template <typename T>
    static void copy_device_to_host( T* h, T* d,  unsigned num_items);

    template <typename T>
    static void copy_device_to_host_and_free( T* h, T* d,  unsigned num_items);

    template <typename T>
    static void copy_host_to_device( T* d, const T* h,  unsigned num_items);


    static void ConfigureLaunch16( dim3& numBlocks, dim3& threadsPerBlock );
    static void ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );
    static void ConfigureLaunch2D( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );
    static void ConfigureLaunch1D( dim3& numBlocks, dim3& threadsPerBlock, unsigned num, unsigned threads_per_block ); 

    static std::string Desc(const dim3& d, int w); 
    static std::string DescLaunch( const dim3& numBlocks, const dim3& threadsPerBlock ); 



};

