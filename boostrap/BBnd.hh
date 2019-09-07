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
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

struct BRAP_API BBnd
{
    static const char DELIM ; 
    static const char* DuplicateOuterMaterial( const char* boundary0 ) ; 
    static const char* Form(const char* omat_, const char* osur_, const char* isur_, const char* imat_);

    BBnd(const char* spec);
    std::string desc() const ;

    const char* omat ; 
    const char* osur ; 
    const char* isur ; 
    const char* imat ; 

    std::vector<std::string> elem ;      
};

#include "BRAP_TAIL.hh"





