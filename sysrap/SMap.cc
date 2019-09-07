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

#include <cassert>
#include <cmath>
#include <iostream>

#include "SMap.hh"
#include "PLOG.hh"


template <typename K, typename V>
unsigned SMap<K,V>::ValueCount(const std::map<K,V>& m, V value)
{
    unsigned count(0); 
    for(typename MKV::const_iterator it=m.begin() ; it != m.end() ; it++)
    {
        V v = it->second ; 
        if( v == value ) count++ ; 
    }
    return count ; 
}


template <typename K, typename V>
void SMap<K,V>::FindKeys(const std::map<K,V>& m, std::vector<K>& keys, V value, bool dump)
{
    if(dump)
    {
        LOG(info) << " value " << std::setw(32) << std::hex << value << std::dec ; 
    } 

    for(typename MKV::const_iterator it=m.begin() ; it != m.end() ; it++)
    {
        K k = it->first ; 
        V v = it->second ; 
        bool match = v == value ; 

        if(dump)
        {
            LOG(info) 
                 << " k " << k 
                 << " v " << std::setw(32) << std::hex << v << std::dec 
                 << " match " << ( match ? "Y" : "N" )
                 << " keys.size() " << keys.size()
                 ;
        } 

        if( match ) keys.push_back(k) ; 
    }
}







template struct SMap<std::string, unsigned long long>;


