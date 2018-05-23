#include <iostream>
#include <iomanip>
#include <vector>

#include "NGLMCF.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"


NGLMCF::NGLMCF( const glm::mat4& A_, const glm::mat4& B_ ) 
       :
        A(A_),
        B(B_),
        epsilon_translation(1e-3f),
        epsilon(1e-5), 
        diff(nglmext::compDiff(A,B)),
        diff2(nglmext::compDiff2(A,B,false,epsilon,epsilon_translation)),
        diffFractional(nglmext::compDiff2(A,B,true,epsilon,epsilon_translation)),
        diffFractionalMax(1e-3),
        match(diffFractional < diffFractionalMax)
{
}

std::string NGLMCF::desc( const char* msg )
{
    std::stringstream ss ; 
    ss <<  msg
       << " epsilon " << epsilon
       << " diff " << diff 
       << " diff2 " << diff2 
       << " diffFractional " << diffFractional
       << " diffFractionalMax " << diffFractionalMax
       << std::endl << gpresent("A", A)
       << std::endl << gpresent("B ",B)
       << std::endl ; 
    
    for(unsigned i=0 ; i < 4 ; i++)
    {
        for(unsigned j=0 ; j < 4 ; j++)
        {
            float a = A[i][j] ;
            float b = B[i][j] ;

            float da = nglmext::compDiff2(a,b, false, epsilon);
            float df = nglmext::compDiff2(a,b, true , epsilon);

            bool ijmatch = df < diffFractionalMax ;

            ss << "[" 
                      << ( ijmatch ? "" : "**" ) 
                      << std::setw(10) << a
                      << ":"
                      << std::setw(10) << b
                      << ":"
                      << std::setw(10) << da
                      << ":"
                      << std::setw(10) << df
                      << ( ijmatch ? "" : "**" ) 
                      << "]"
                       ;
        }
        ss << std::endl; 
    }

    return ss.str();
}

