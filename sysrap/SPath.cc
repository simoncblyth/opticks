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
#include <iostream>

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

const char* SPath::Dirname(const char* path)
{
    std::string p = path ; 
    std::size_t pos = p.find_last_of("/");
    std::string dir = pos == std::string::npos ? p : p.substr(0,pos) ; 
    return strdup( dir.c_str() ) ; 
}

const char* SPath::Basename(const char* path)
{
    std::string p = path ; 
    std::size_t pos = p.find_last_of("/");
    std::string base = pos == std::string::npos ? p : p.substr(pos+1) ; 
    return strdup( base.c_str() ) ; 
}

const char* SPath::UserTmpDir(const char* pfx, const char* user_envvar, const char* sub, char sep  ) // static 
{
    char* user = getenv(user_envvar); 
    std::stringstream ss ; 
    ss << pfx 
       << sep
       << user 
       << sep
       << sub 
       ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}

/**
SPath::Resolve
---------------

Resolves tokenized paths such as "$PREFIX/name.ext" where PREFIX must 
be an existing envvar. Special handling if "$TMP" is provided that defaults 
the TMP envvar to "/tmp/username/opticks" 

**/

const char* SPath::Resolve(const char* spec_)
{
    if(!spec_) return NULL ;       

    char* spec = strdup(spec_);            // copy to allow modifications 
    char sep = '/' ; 
    char* spec_sep = strchr(spec, sep);    // pointer to first separator
    char* spec_end = strchr(spec, '\0') ;  // pointer to null terminator

    std::stringstream ss ; 
    if(spec[0] == '$' && spec_sep && spec_end && spec_sep != spec_end)
    {
        *spec_sep = '\0' ; // temporarily null terminate at the first slash  
        char* pfx = getenv(spec+1); 
        *spec_sep = sep ;  // put back the separator
        const char* prefix = pfx ? pfx : UserTmpDir() ; 
        ss << prefix << spec_sep ; 
    }
    else
    {
        ss << spec ; 
    }
    std::string s = ss.str(); 
    return strdup(s.c_str()) ; 
}





