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
#include "SSys.hh"
#include "PLOG.hh"


const plog::Severity NGLMCF::LEVEL = PLOG::EnvLevel("NGLMCF", "debug") ; 


NGLMCF::NGLMCF( const glm::mat4& A_, const glm::mat4& B_ ) 
    :
    A(A_),
    B(B_),
    diff(nglmext::compDiff(A,B)),
    diff2(nglmext::compDiff2(A,B,false)),
    diffFractional(nglmext::compDiff2(A,B,true)),
    diffFractionalCheck(nglmext::compDiff2_check(A,B,true)), 
    diffFractionalMax(1e-3),
    diffMax(epsilon_translation*10.),
    match(diffFractional < diffFractionalMax)
{
    if(SSys::getenvbool("NGLMCF")) LOG(LEVEL) << desc("NGLMCF"); 
}


template<typename T>
const plog::Severity NGLMCF_<T>::LEVEL = PLOG::EnvLevel("NGLMCF_", "debug") ; 


template<typename T>
NGLMCF_<T>::NGLMCF_( const glm::tmat4x4<T>& A_, const glm::tmat4x4<T>& B_ ) 
    :
    A(A_),
    B(B_),
    diff(nglmext::compDiff_(A,B)),
    diff2(nglmext::compDiff2_(A,B,false)),
    diffFractional(nglmext::compDiff2_(A,B,true)),
    diffFractionalCheck(nglmext::compDiff2_check_(A,B,true)),
    diffFractionalMax(T(1e-3)),
    diffMax(epsilon_translation*10.),
    match(diffFractional < diffFractionalMax)
{
    if(SSys::getenvbool("NGLMCF_")) LOG(LEVEL) << desc("NGLMCF_"); 
}

/**

Issues with elementwise comparison of mat4 are:

1. absolute comparisons of large translation values 
2. fractional comparisons of values very close to zero  

   * to address this any values under epsilon are set to zero 
     in the comparison 

**/

std::string NGLMCF::desc( const char* msg, int width )
{
    std::stringstream ss ; 
    ss <<  msg
       << " epsilon " << epsilon
       << " epsilon_translation " << epsilon_translation
       << std::endl 
       << " diff " << diff 
       << " diff2 " << diff2 
       << std::endl 
       << " diffFractional " << ( match ? "" : "[*" ) << diffFractional << ( match ? "" : "*]" )
       << " diffFractionalMax " << diffFractionalMax
       << std::endl 
       << std::endl << gpresent("A", A)
       << std::endl << gpresent("B ",B)
       << std::endl << gpresent("DFC",diffFractionalCheck)
       << std::endl ; 
    

    for(unsigned mode=0 ; mode < 2 ; mode++)
    {
        bool fractional = mode == 0 ? false : true ; 
        float dmax = fractional ? diffFractionalMax : diffMax ; 

        switch(mode)
        {
            case 0: ss << "mode 0: absolute difference " ; break ; 
            case 1: ss << "mode 1: fractional : absolute difference/average  " ; break ; 
        }
        ss << std::endl ; 

        for(unsigned i=0 ; i < 4 ; i++)
        {
            for(unsigned j=0 ; j < 4 ; j++)
            {
                float a = A[i][j] ;
                float b = B[i][j] ;
                float u_epsilon = i == 3 ? epsilon_translation : epsilon ; 
                float d  = nglmext::compDiff2(a,b, fractional, u_epsilon); 

                bool ijmatch = d < dmax ;

                ss << "[" 
                          << std::setw(2) << ( ijmatch ? "" : "**" ) 
                          << std::setw(width) << a
                          << ":"
                          << std::setw(width) << b
                          << ":"
                          << std::setw(width) << d
                          << ":"
                          << std::setw(2) << ( ijmatch ? "" : "**" ) 
                          << "]"
                           ;
            }
            ss << std::endl; 
        }
    } 

    return ss.str();
}



template<typename T>
std::string NGLMCF_<T>::desc( const char* msg, int width )
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
                      << std::setw(width) << a
                      << ":"
                      << std::setw(width) << b
                      << ":"
                      << std::setw(width) << da
                      << ":"
                      << std::setw(width) << df
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

