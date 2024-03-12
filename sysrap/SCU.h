#pragma once
#include <cstdint>
#include <vector_types.h>
#include <ostream>
#include <iomanip>

struct SCU
{
    static void ConfigureLaunch2D( dim3& numBlocks, dim3& threadsPerBlock, int32_t width, int32_t height ); 
};

inline void SCU::ConfigureLaunch2D( dim3& numBlocks, dim3& threadsPerBlock, int32_t width, int32_t height ) // static
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 16 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}

inline std::ostream& operator<<(std::ostream& os, const dim3& v)
{
    int w = 6 ;
    os
       << "("
       << std::setw(w) << v.x
       << ","
       << std::setw(w) << v.y
       << ","
       << std::setw(w) << v.z
       << ") "
       ;
    return os;
}

