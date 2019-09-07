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

#include <string>
#include <vector>


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"
#include "NYJSON.hpp"

/**
NJS
====

Convenience wrapper for JSON read/write 

**/


class NPY_API NJS {
   public:
       NJS(); 
       NJS(const NJS& other); 
       NJS(const nlohmann::json& js ); 
   public:
       nlohmann::json& js();
       const nlohmann::json& cjs() const ;
       void read(const char* path0, const char* path1=NULL);
       void write(const char* path0, const char* path1=NULL) const ;
       void dump(const char* msg="NJS::dump") const ; 
   private:
       nlohmann::json  m_js ;  
};

#include "NPY_TAIL.hh"

