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
SPairVec<K,V>
===============

Utilities for sorting vectors of key value pairs.


**/


#include <map>
#include <vector>

#include "SYSRAP_API_EXPORT.hh"
 
template <typename K, typename V>
struct SYSRAP_API SPairVec
{
    typedef typename std::pair<K,V>    PKV ; 
    typedef typename std::vector<PKV>  LPKV ; 

    SPairVec( LPKV& lpkv, bool ascending ); 

    bool operator()(const PKV& a, const PKV& b);
    void sort(); 
    void dump(const char* msg) const ; 

    LPKV&  _lpkv ; 
    bool   _ascending ; 

};


