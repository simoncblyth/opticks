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
}



void test_vector_erase_all()
{
    LOG(info); 

    std::vector<int> a = {0,1,2,3,4,5,6,7,8,9} ;

    std::vector<int>::const_iterator beg = a.cbegin(); 
    std::vector<int>::const_iterator end = a.cend(); 

    a.erase(beg, end); 

    LOG(info) << " a: " << a.size(); 
    for(unsigned i=0 ; i < a.size() ; i++) std::cout << a[i] << " " ; 
}







int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_MaxDiff();
    //test_FindIndexOfValue();
    test_vector_erase_pos(); 
    test_vector_erase_all(); 

    return 0 ;
}

// om-;TEST=SVecTest om-t 
