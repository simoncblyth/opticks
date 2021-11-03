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
#include <cstring>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <sstream>

#include "SStr.hh"
#include "SVec.hh"


template <typename T>
void SVec<T>::Dump( const char* label, const std::vector<T>& a  )
{
    std::cout << std::setw(10) << label  ;
    for(unsigned i=0 ; i < a.size() ; i++) std::cout << std::setw(10) << a[i] << " " ; 
    std::cout << std::endl ; 
} 

template <typename T>
std::string SVec<T>::Desc( const char* label, const std::vector<T>& a, int width  )
{
    std::stringstream ss ; 
    ss << std::setw(10) << label  ;
    for(unsigned i=0 ; i < a.size() ; i++) ss << std::setw(width) << a[i] << " " ; 
    return ss.str(); 
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
        T df = a[i] - b[i] ;   // std::abs ambiguous when T=unsigned 
        if( df < 0 ) df = -df ;  

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
        //T df = std::abs(a[i] - value) ;   // std::abs ambiguous when T=unsigned 
        T df = a[i] - value ; 
        if(df < 0) df = -df ; 

        if(df < tolerance)
        {   
            idx = i ; 
            break ; 
        }   
    }           
    return idx ; 
}


template <typename T>
int SVec<T>::FindIndexOfValue(const std::vector<T>& a, T value )
{
    size_t idx = std::distance( a.begin(), std::find( a.begin(), a.end(), value )) ; 
    return idx < a.size() ? idx : -1 ; 
}
 


template <typename T>
void SVec<T>::MinMaxAvg(const std::vector<T>& t, T& mn, T& mx, T& av) 
{
    typedef typename std::vector<T>::const_iterator IT ;    
    IT mn_ = std::min_element( t.begin(), t.end()  );  
    IT mx_ = std::max_element( t.begin(), t.end()  );  
    double sum = std::accumulate(t.begin(), t.end(), T(0.) );   

    mn = *mn_ ; 
    mx = *mx_ ; 
    av = t.size() > 0 ? sum/T(t.size()) : T(-1.) ;   
}

template <typename T>
void SVec<T>::MinMax(const std::vector<T>& t, T& mn, T& mx ) 
{
    typedef typename std::vector<T>::const_iterator IT ;    
    IT mn_ = std::min_element( t.begin(), t.end()  );  
    IT mx_ = std::max_element( t.begin(), t.end()  );  
    mn = *mn_ ; 
    mx = *mx_ ; 
}

template <typename T>
void SVec<T>::Extract(std::vector<T>& a, const char* str0, const char* ignore ) 
{
    char swap = ' '; 
    const char* str1 = SStr::ReplaceChars(str0, ignore, swap); 
    std::stringstream ss(str1);  
    std::string s ; 
    T value ; 

    while(std::getline(ss, s, ' '))
    {
        if(strlen(s.c_str()) == 0 ) continue;  
        //std::cout << "[" << s << "]" << std::endl ; 
        
        std::stringstream tt(s);  
        tt >> value ; 
        a.push_back(value); 
    }
}


template struct SVec<int>;
template struct SVec<unsigned>;
template struct SVec<float>;
template struct SVec<double>;


