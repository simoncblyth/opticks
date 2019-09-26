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

#include "NPY_API_EXPORT.hh"

/**
NTreeAnalyse
==============

* see ../analytic/csg.py 

**/

#include <string>
template <typename T> struct NNodeCollector ; 
template <typename T> struct NGrid ; 

template <typename T>
struct NPY_API NTreeAnalyse
{
    static std::string Desc(const T* root_);
    static std::string Brief(const T* root_);
  
    NTreeAnalyse(const T* root_); 
    ~NTreeAnalyse(); 

    void init(); 
    void initGrid(); 
    unsigned depth_(bool label);
    unsigned depth_r(const T* node, unsigned depth, bool label);
    std::string desc() const ;
    std::string brief() const ;

    const T*           root ; 
    unsigned           height ; 
    NNodeCollector<T>* nodes ; 
    unsigned           count ; 
    NGrid<T>*          grid ; 
 

};



