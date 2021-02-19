/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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


template<typename T>
NGLMCF_<T>::NGLMCF_( const glm::tmat4x4<T>& A_, const glm::tmat4x4<T>& B_ ) 
    :
    A(A_),
    B(B_),
    epsilon_translation(T(1e-3)),
    epsilon(T(1e-5)), 
    diff(nglmext::compDiff_(A,B)),
    diff2(nglmext::compDiff2_(A,B,false,epsilon,epsilon_translation)),
    diffFractional(nglmext::compDiff2_(A,B,true,epsilon,epsilon_translation)),
    diffFractionalMax(T(1e-3)),
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



template<typename T>
std::string NGLMCF_<T>::desc( const char* msg )
{
    std::stringstream ss ; 
    ss <<  msg
       << " epsilon " << epsilon
       << " diff " << diff 
       << " diff2 " << diff2 
       << " diffFractional " << diffFractional
       << " diffFractionalMax " << diffFractionalMax
       << std::endl << gpresent__("A", A)
       << std::endl << gpresent__("B ",B)
       << std::endl ; 
    
    for(unsigned i=0 ; i < 4 ; i++)
    {
        for(unsigned j=0 ; j < 4 ; j++)
        {
            T a = A[i][j] ;
            T b = B[i][j] ;

            T da = nglmext::compDiff2_(a,b, false, epsilon);
            T df = nglmext::compDiff2_(a,b, true , epsilon);

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



template struct NGLMCF_<float> ; 
template struct NGLMCF_<double> ; 

