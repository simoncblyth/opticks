/**
SCU_test.cc
============

::

    ~/o/sysrap/tests/SCU_test.sh 


**/
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>

#include "SCU.h"

struct SCU_test
{
    static int ConfigureLaunch2D(); 

    template<typename T>
    static void FillVec( std::vector<T>& vec, int N );
 
    template<typename T>
    static int Compare( const T* arr0, const T* arr1, size_t num_item );
 
   static int UploadArray_DownloadArray();
    static int UploadBuf_DownloadBuf();
    static int Buf();
    static int FreeBuf();
}; 


inline int SCU_test::ConfigureLaunch2D()
{
    std::cout << "[ SCU_test::ConfigureLaunch2D " << std::endl ;
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 

    int32_t width = 1024 ; 
    int32_t height = 768 ; 
    SCU::ConfigureLaunch2D(numBlocks, threadsPerBlock, width, height );   

    std::cout 
        << " width " << width
        << " height " << height
        << " numBlocks " << numBlocks   
        << " threadsPerBlock " << threadsPerBlock
        << std::endl
        ;
    std::cout << "] SCU_test::ConfigureLaunch2D " << std::endl ;
    return 0 ; 
}


template<typename T>
inline void SCU_test::FillVec( std::vector<T>& vec, int N  )
{
    vec.resize(N); 
    for(int i=0 ; i < N ; i++ ) vec[i] = T(i) ; 
}


template<typename T>
inline int SCU_test::Compare( const T* arr0, const T* arr1, size_t num_item )
{
    int deviant = 0 ; 
    for(size_t i=0 ; i < num_item ; i++)
    {
        bool expect = arr0[i] == arr1[i] ; 
        if(!expect) deviant += 1 ;  
        if(!expect || i % 100 == 0)
        std::cout 
           << "  i " << std::setw(6) << i 
           << " arr0 " << std::setw(10) << std::setprecision(4) << std::fixed << arr0[i] 
           << " arr1 " << std::setw(10) << std::setprecision(4) << std::fixed << arr1[i]
           << std::endl 
           ; 
        assert(expect); 
    } 
    return deviant ; 
}


inline int SCU_test::UploadArray_DownloadArray()
{
    std::cout << "[ SCU_test::UploadArray_DownloadArray " << std::endl ;

    std::vector<float> vec ;
    FillVec<float>(vec, 1000); 

    const float* arr0 = vec.data() ;  
    size_t num_item = vec.size() ; 

    float* d_arr = SCU::UploadArray<float>( arr0, num_item ); 
    const float* arr1 = SCU::DownloadArray<float>( d_arr, num_item ); 

    int deviant = Compare( arr0, arr1, num_item );  

    std::cout << "] SCU_test::UploadArray_DownloadArray " << std::endl ;
    return deviant ; 
}


inline int SCU_test::UploadBuf_DownloadBuf()
{
    std::cout << "[ SCU_test::UploadBuf_DownloadBuf " << std::endl ;

    std::vector<float> vec ;
    FillVec<float>(vec, 1000); 

    const float* arr0 = vec.data() ;  
    size_t num_item = vec.size() ; 

    SCU_Buf<float> buf = SCU::UploadBuf<float>( arr0, num_item, "arr0" ); 
    const float* arr1 = SCU::DownloadBuf<float>( buf ); 
    std::cout << " buf.desc " << buf.desc() << std::endl ;


    int deviant = Compare( arr0, arr1, num_item );  
    std::cout << "] SCU_test::UploadBuf_DownloadBuf " << std::endl ;
    return deviant ; 
}

inline int SCU_test::Buf()
{
    SCU_Buf<float> buf = {} ; 
    std::cout << " buf.desc " << buf.desc() << std::endl ;
    return 0 ; 
}

inline int SCU_test::FreeBuf()
{
    std::vector<float> vec ;
    FillVec(vec, 1000); 

    SCU_Buf<float> buf = SCU::UploadBuf<float>( vec.data(), vec.size(), "arr0" ); 
    std::cout << "SCU_test::FreeBuf.0  buf.desc " << buf.desc() << std::endl ;

    //SCU::FreeBuf(buf) ; 
    buf.free(); 

    std::cout << "SCU_test::FreeBuf.1  buf.desc " << buf.desc() << std::endl ;

    return 0 ; 
}


int main()
{
    int rc = 0 ; 
    rc += SCU_test::FreeBuf(); 
    /*
    rc += SCU_test::Buf(); 
    rc += SCU_test::UploadBuf_DownloadBuf(); 
    rc += SCU_test::UploadArray_DownloadArray(); 
    rc += SCU_test::ConfigureLaunch2D(); 
    */
    return rc ; 
}
