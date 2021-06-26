// om-;TEST=SVecTest om-t 
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

#include <cmath>
#include <algorithm>
#include "SVec.hh"

#include "OPTICKS_LOG.hh"


void test_MaxDiff()
{
    float epsilon = 1e-5 ; 

    std::vector<float> a = {1,2,3,4} ;
    std::vector<float> b = {1.1,2.2,3.3,4.4} ;

    bool dump = true ; 
    float md = SVec<float>::MaxDiff(a, b, dump) ; 
    float md_x = 0.4f ; 
    float md_d = std::fabs(md - md_x) ; 

    assert( md_d < epsilon );
}

void test_FindIndexOfValue()
{
    std::vector<float> a = {1.1,2.2,3.3,4.4} ;
    int idx ;  

    idx = SVec<float>::FindIndexOfValue( a, 3.3f, 1e-6f );
    assert( idx == 2 );

    idx = SVec<float>::FindIndexOfValue( a, 5.5f, 1e-6f );
    assert( idx == -1 );
}


void test_vector_erase_pos()
{
    LOG(info); 

    std::vector<int> a = {0,1,2,3,4,5,6,7,8,9} ;

    auto pos = std::find(std::begin(a), std::end(a), 5);

    a.erase(pos); 

    LOG(info) << " a: " << a.size(); 
    for(unsigned i=0 ; i < a.size() ; i++) std::cout << a[i] << " " ; 
    std::cout << std::endl ; 
}




// clang comes up with (4 * 10000 + 2 * 100 + 1)

#define GCC_VERSION (__GNUC__ * 10000 \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)


#define XSTR(x) STR(x)
#define STR(x) #x
//#pragma message "GCC_VERSION = " XSTR(GCC_VERSION) 


void test_vector_erase_all()
{
    LOG(info); 

    const char* gcc_version = XSTR(GCC_VERSION) ;
    LOG(info) << " GCC_VERSION : " << gcc_version ;  

    std::vector<int> a = {0,1,2,3,4,5,6,7,8,9} ;

    // GCC_VERSION cut is guess based on what Geant4 1062 needs
#if GCC_VERSION > 40903 || __clang__  

    std::vector<int>::const_iterator beg = a.cbegin(); 
    std::vector<int>::const_iterator end = a.cend(); 

    a.erase(beg, end); 
#else
    LOG(fatal) << " needs newer GCC_VERSION than : " << gcc_version ; 
#endif

    LOG(info) << " a: " << a.size(); 
    for(unsigned i=0 ; i < a.size() ; i++) std::cout << a[i] << " " ; 
    std::cout << std::endl ; 
}

void test_unique_strings()
{
    std::vector<std::string> v = { "red", "green", "blue", "cyan", "magenta", "yellow", "green" } ; 
    std::vector<std::string> u ; 

    for(unsigned i=0 ; i < v.size() ; i++)
    {
        const std::string& s = v[i] ;  
        if(std::find(u.begin(), u.end(), s ) == u.end()) u.push_back(s); 
    }

    for(unsigned i=0 ; i < u.size() ; i++) std::cout << u[i] << std::endl ; 
}


void test_MinMaxAvg()
{
    std::vector<float> v = {1.f, 10.f, 100.f, 2.f, 1000.f } ;  
    float mn, mx, av ; 
    SVec<float>::MinMaxAvg(v,mn,mx,av);

    LOG(info) 
        << " mn " << mn 
        << " mx " << mx 
        << " av " << av 
        ;

}


void test_Desc()
{
     std::vector<int> v = { 380, 400, 420, 440, 460 } ; 
     LOG(info) << SVec<int>::Desc("v", v); 
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_MaxDiff();
    //test_FindIndexOfValue();
    //test_vector_erase_pos(); 
    //test_vector_erase_all(); 
    //test_unique_strings(); 
    //test_MinMaxAvg(); 

    test_Desc();  

    return 0 ;
}

// om-;TEST=SVecTest om-t 
