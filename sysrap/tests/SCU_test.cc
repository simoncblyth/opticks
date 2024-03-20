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
    static int UploadArray_DownloadArray();
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

inline int SCU_test::UploadArray_DownloadArray()
{
    std::cout << "[ SCU_test::UploadArray_DownloadArray " << std::endl ;
    static const int N = 1000 ; 
    std::vector<float> vec(N) ;  
    for(int i=0 ; i < N ; i++ ) vec[i] = float(i) ; 

    const float* arr0 = vec.data() ;  
    unsigned num_items = vec.size() ; 

    float* d_arr = SCU::UploadArray<float>( arr0, num_items ); 
    const float* arr1 = SCU::DownloadArray<float>( d_arr, num_items ); 

    int deviant = 0 ; 

    for(unsigned i=0 ; i < num_items ; i++)
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
    std::cout << "] SCU_test::UploadArray_DownloadArray " << std::endl ;
    return deviant ; 
}


int main()
{
    int rc = 0 ; 
    rc += SCU_test::UploadArray_DownloadArray(); 
    rc += SCU_test::ConfigureLaunch2D(); 
    return rc ; 
}
