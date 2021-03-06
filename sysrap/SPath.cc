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

#include <string>
#include <cstring>
#include <cassert>
#include <sstream>
#include <fstream>

#include "SPath.hh"

const char* SPath::Stem( const char* name ) // static
{
    std::string arg = name ;
    std::string base = arg.substr(0, arg.find_last_of(".")) ; 
    return strdup( base.c_str() ) ; 
}

bool SPath::IsReadable(const char* path)  // static 
{
    std::ifstream fp(path, std::ios::in|std::ios::binary);
    bool readable = !fp.fail(); 
    fp.close(); 
    return readable ; 
}

const char* SPath::GetHomePath(const char* rel)  // static 
{
    char* home = getenv("HOME"); 
    assert(home);  
    std::stringstream ss ; 
    ss << home ;
    if(rel != NULL) ss << "/" << rel ; 
    std::string path = ss.str(); 
    return strdup(path.c_str()) ; 
}



