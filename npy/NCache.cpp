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

#include "NCache.hpp"
#include <cstdio>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


NCache::NCache(const char* dir) 
   : 
      m_cache(dir) 
{
} 


std::string NCache::path(const char* relative)
{   
    fs::path cpath(m_cache/relative); 
    return cpath.string();
} 

std::string NCache::path(const char* tmpl, const char* incl)
{   
    char p[128];
    snprintf(p, 128, tmpl, incl);
    fs::path cpath(m_cache/p); 
    return cpath.string();
}   

std::string NCache::path(const char* tmpl, unsigned int incl)
{   
    char p[128];
    snprintf(p, 128, tmpl, incl);
    fs::path cpath(m_cache/p); 
    return cpath.string();
} 


 
