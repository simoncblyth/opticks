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

#include <sstream>
#include <string>
#include "STime.hh"
#include "SPath.hh"
#include <time.h>

#include "s_time.h"


std::string STime::mtime(const char* base, const char* name)
{
    std::stringstream ss ;  
    ss << base << "/" << name ; 
    std::string str = ss.str(); 
    return mtime(str.c_str());  
}

std::string STime::mtime(const char* path)
{
    int mt = SPath::mtime(path);   
    return mt > 0 ? s_time::Format(mt, nullptr) : "" ;
}





