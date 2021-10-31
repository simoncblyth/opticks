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
#include "plog/Severity.h"
#include <vector>

/**
NTreeBalance
==============

NB this is **NOT A GENERAL TREE BALANCER** it does 
however succeed to balance trees that Geant4 
boolean solids often result in.

Ported from python:

* ../analytic/csg.py 
* ../analytic/treebuilder.py

**/

template <typename T>
struct NPY_API NTreeBalance
{
    static const plog::Severity LEVEL ; 

    NTreeBalance(T* root_, bool dump_); 

    T* create_balanced(); 

    static bool is_collected(const std::vector<T*>& subs, const T* node);

    void init(); 
    static unsigned depth_r(T* node, unsigned depth, bool label);
    static void     subdepth_r(T* node, unsigned depth );

    unsigned operators(unsigned minsubdepth=0) const ;
    std::string operatorsDesc(unsigned minsubdepth=0) const ;

    static void operators_r(const T* node, unsigned& mask, unsigned minsubdepth );

    void subtrees(std::vector<T*>& subs, unsigned subdepth, std::vector<T*>& otherprim );
    static void subtrees_r(T* node, std::vector<T*>& subs, unsigned subdepth, std::vector<T*>& otherprim, unsigned pass );

    bool is_positive_form() const ;  
    bool is_mono_form()     const ;  
    bool is_unable_to_balance() const ; 

    T*           root ; 
    unsigned     height0 ; 
    bool         unable_to_balance ; 
    bool         dump ; 

};



