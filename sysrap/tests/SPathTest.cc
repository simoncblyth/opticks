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

// om-;TEST=SPathTest om-t 

#include <cassert>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>

#include "SPath.hh"

#include "OPTICKS_LOG.hh"


void test_Stem()
{
    LOG(info); 
    const char* name = "hello.cu" ; 
    const char* stem = SPath::Stem(name); 
    const char* x_stem = "hello" ; 
    assert( strcmp( stem, x_stem ) == 0 ); 
}

void test_GetHomePath()
{
    LOG(info); 
    const char* bashrc = SPath::GetHomePath(".bashrc") ; 
    std::cout << bashrc << std::endl ; 
}

void test_IsReadable()
{
    LOG(info); 
    const char* self = SPath::GetHomePath("opticks/sysrap/tests/SPathTest.cc") ; 
    const char* non = SPath::GetHomePath("opticks/sysrap/tests/SPathTest.cc.non") ; 
    std::cout << self << std::endl ; 
    bool readable = SPath::IsReadable(self); 
    assert( readable == true ); 

    bool readable_non = SPath::IsReadable(non); 
    assert( readable_non == false ); 
}

void test_Basename()
{
    LOG(info); 
    std::vector<std::string> paths = { "/dd/materials/Water", "Water", "" } ; 
    for(unsigned i=0 ; i < paths.size() ; i++)
    {
        const char* path = paths[i].c_str() ;
        const char* base = SPath::Basename(path) ;
        std::cout 
            << " path [" << path << "]"  
            << " base [" << base << "]" 
            << std::endl 
            ;  
    }
}


int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    test_Stem();  
    test_GetHomePath();  
    test_IsReadable();  
    test_Basename(); 

    return 0  ; 
}
// om-;TEST=SPathTest om-t 
