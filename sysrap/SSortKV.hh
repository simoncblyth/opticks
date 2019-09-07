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

#pragma once

/**
SSortKV
==========

Utility struct for reordering a vector of string float pairs.

**/


#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include "SYSRAP_API_EXPORT.hh"
 
struct SYSRAP_API SSortKV 
{
    typedef std::pair<std::string, float> KV ; 
    typedef std::vector<KV> VKV ;

    SSortKV(bool descending_) : descending(descending_) {} 

    void add(const char* k, float v)
    {    
        vkv.push_back(KV(k, v));
    }    
    void sort()
    {    
        std::sort(vkv.begin(), vkv.end(), *this );
    }    
    void dump(const char* msg="SSortKV::dump") const ;

    bool operator()( const KV& a, const KV& b) const 
    {    
        return descending ? a.second > b.second : a.second < b.second ; 
    }    
    unsigned getNum() const  
    {
        return vkv.size();
    }
    const std::string& getKey(unsigned i) const 
    {    
        return vkv[i].first ; 
    }    
    float getVal(unsigned i) const 
    {    
        return vkv[i].second ; 
    }    

    VKV   vkv ; 
    bool  descending ;  

};

