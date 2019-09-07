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
#include <vector>
#include <string>

/**
NNodeCollector
===============

Works with node structs with members: 

* left, right pointing to other nodes
* char* label  

Trying to stay const correct, makes difficult to 
use in NTreeAnalyse


**/

template <typename T>
struct NPY_API NNodeCollector
{
    static void Inorder_r( std::vector<T*>& inorder, T* node ); 

    NNodeCollector( const T* root_ ); 
    ~NNodeCollector(); 

    void collect_preorder_r( const T* node ); 
    void collect_inorder_r( const T* node ); 
    void collect_postorder_r( const T* node ); 

    void dump( const char* msg, std::vector<const T*>& order ) ;
    void dump( const char* msg="NNodeCollector::dump" ) ;

    std::string desc( std::vector<const T*>& order ) ;
    std::string desc_inorder() ;
    std::string desc_preorder() ;
    std::string desc_postorder() ;


    const T* root ; 
    std::vector<const T*> inorder ; 
    std::vector<const T*> preorder ; 
    std::vector<const T*> postorder ; 

};


