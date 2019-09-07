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

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <iterator>

#include "SVec.hh"


template <typename T>
void SVec<T>::Dump( const char* label, const std::vector<T>& a  )
{
    std::cout << std::setw(10) << label  ;
    for(unsigned i=0 ; i < a.size() ; i++) std::cout << std::setw(10) << a[i] << " " ; 
    std::cout << std::endl ; 
} 


template <typename T>
void SVec<T>::Dump2( const char* label, const std::vector<T>& a  )
{
    std::cout << std::setw(10) << label ;
    std::copy( a.begin(), a.end(), std::ostream_iterator<float>(std::cout, " ")) ;
    std::cout << std::endl ; 
} 

template <typename T>
T SVec<T>::MaxDiff(const std::vector<T>& a, const std::vector<T>& b, bool dump)
{
    assert( a.size() == b.size() );
    T mx = 0.f ;     
    for(unsigned i=0 ; i < a.size() ; i++)
    {
        T df = std::abs(a[i] - b[i]) ; 
        if(df > mx) mx = df ; 

        if(dump)
        std::cout 
            << " a " << a[i] 
            << " b " << b[i] 
            << " df " << df
            << " mx " << mx 
            << std::endl 
            ; 

    }
    return mx ; 
}


template <typename T>
int SVec<T>::FindIndexOfValue(const std::vector<T>& a, T value, T tolerance)
{
    int idx = -1 ; 
    for(unsigned i=0 ; i < a.size() ; i++)
    {   
        T df = std::abs(a[i] - value) ; 
        if(df < tolerance)
        {   
            idx = i ; 
            break ; 
        }   
    }           
    return idx ; 
}
 



template struct SVec<float>;
template struct SVec<double>;


