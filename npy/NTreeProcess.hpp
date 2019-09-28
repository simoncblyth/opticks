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

#include <vector>
#include "NPY_API_EXPORT.hh"

/**
NTreeProcess
==============

**/

template <typename T> class  NPY ; 
template <typename T> class  NTreePositive ; 
template <typename T> struct NTreeBalance ; 

template <typename T>
struct NPY_API NTreeProcess
{
    static unsigned MaxHeight0 ;  
    static T* Process( T* root_ , unsigned soIdx, unsigned lvIdx );
    static std::vector<unsigned>*  LVList ;  
    static NPY<unsigned>* ProcBuffer ; 
    static void SaveBuffer(const char* path) ; 
    static void SaveBuffer(const char* dir, const char* name) ; 

    NTreeProcess(T* root_); 
    void init();

    T* root ; 
    T* balanced  ; 
    T* result  ; 

    NTreeBalance<T>*  balancer ; 
    NTreePositive<T>* positiver ; 

};



