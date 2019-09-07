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

#include <algorithm>
#include <iostream>

#include "SPairVec.hh"
#include "PLOG.hh"


template <typename K, typename V>
SPairVec<K,V>::SPairVec( LPKV& lpkv, bool ascending )
   :
   _lpkv(lpkv), 
   _ascending(ascending)
{
}

template <typename K, typename V>
bool SPairVec<K,V>::operator()(const PKV& a, const PKV& b)
{
    return _ascending ? a.second < b.second : a.second > b.second ; 
}

template <typename K, typename V>
void SPairVec<K,V>::sort()
{
    std::sort( _lpkv.begin(), _lpkv.end(), *this );
}


template <typename K, typename V>
void SPairVec<K,V>::dump(const char* msg) const 
{
    LOG(info) << msg << " size " << _lpkv.size() ; 
    for(unsigned i=0 ; i < _lpkv.size() ; i++)
    {
        std::cerr 
            << " " << _lpkv[i].first 
            << " " << _lpkv[i].second
            << std::endl 
            ;
    }
}



template struct SPairVec<std::string, unsigned>;


