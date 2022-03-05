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

// TEST=STimeTest om-t

#include <vector>
#include <iostream>

#include "STime.hh"
#include "SSys.hh"

#include "OPTICKS_LOG.hh"


void test_EpochSeconds()
{
    LOG(info); 
    int t = STime::EpochSeconds() ;      

    std::cout << "STime::EpochSeconds() " << t << std::endl ; 
    std::cout << "STime::Format(\"%c\", t) " << STime::Format(t, "%c") << std::endl;
    std::cout << "STime::Format() " << STime::Format() << std::endl;


    std::vector<std::string> fmts = { "%c", "%Y", "%m", "%D", "%d", "%H", "%M", "%Y%m%d_%H%M%S"  } ; 

    for(unsigned i=0 ; i < fmts.size() ; i++)
    {
         std::cout 
               << "STime::Format(0,\"" << fmts[i] << "\") " 
               <<  STime::Format(0, fmts[i].c_str()) 
               << std::endl;
    }



    SSys::run( "date +%s" ); 
}

void test_mtime()
{
    LOG(info); 
    const char* path = "/tmp/tt.txt" ; 
    std::string mt = STime::mtime(path); 
    LOG(info) << " path " << path << " mt " << mt ; 
}


int main(int argc, char** argv) 
{
    OPTICKS_LOG(argc, argv);

    //test_EpochSeconds(); 
    test_mtime(); 

    return 0 ; 
}
