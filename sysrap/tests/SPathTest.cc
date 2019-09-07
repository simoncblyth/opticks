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
#include <string>
#include "SPath.hh"

#include "OPTICKS_LOG.hh"


void test_Stem()
{
    const char* name = "hello.cu" ; 
    const char* stem = SPath::Stem(name); 
    const char* x_stem = "hello" ; 
    assert( strcmp( stem, x_stem ) == 0 ); 
}


int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    test_Stem();  

    return 0  ; 
}

