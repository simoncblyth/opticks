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

// TEST=ArrayTest om-t 

#include "OPTICKS_LOG.hh"

#include <sstream>
#include <string>
#include <cstring>
#include <array>

struct Demo
{
    Demo(const char* name_, int value_)
       :
       name(strdup(name_)),
       value(value_) 
    {
    }
    std::string desc() const 
    {
        std::stringstream ss ; 
        ss
           << std::setw(10) << name
           << " : " 
           << std::setw(10) << value
           ;
        return ss.str();
    }

    const char* name ; 
    int value ; 
};


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    std::array<Demo*, 10> arr ; 
    arr.fill(NULL); 

    arr[5] = new Demo("yo", 42) ; 

    for(unsigned i=0 ; i < arr.size() ; i++) LOG(info) << i << " : " << ( arr[i] ? arr[i]->desc() : "-" ) ; 

    arr[10] = new Demo("hmm", 42) ; 

    return 0 ; 
}


