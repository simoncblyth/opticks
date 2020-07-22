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

#include "NNodeEnum.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

#include "plog/Severity.h"


/*
NTreeChopper
=============

*/

template <typename T> class  NPY ; 

template <typename T>
struct NPY_API NTreeChopper
{
    static const plog::Severity LEVEL ; 

    T* root ; 
    const float epsilon ; 
    const unsigned verbosity ; 
    bool enabled ; 

    std::vector<T*>           prim ; 
    std::vector<nbbox>        bb ; 
    std::vector<nbbox>        cc ; 
    std::vector<unsigned>     zorder ; 

    NTreeChopper(T* root, float epsilon ) ;
  
    void init();
    void update_prim_bb();  // direct from param, often with gtransform applied
    bool operator()( int i, int j)  ;

    unsigned get_num_prim() const ; 
    std::string brief() const ;

    void dump(const char* msg="NTreeChopper::dump");
    void dump_qty(char qty, int wid=10);
    void dump_joins();

};


