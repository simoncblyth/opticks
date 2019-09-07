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
template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"
#include "plog/Severity.h"

class NPY_API NLoad {
       static const plog::Severity LEVEL ; 
   public:
       static NPY<float>* Gensteps( const char* det, const char* typ, const char* tag );
       static std::string GenstepsPath( const char* det, const char* typ, const char* tag );
       static std::string directory(const char* pfx, const char* det, const char* typ, const char* tag, const char* anno=NULL);
       static std::string reldir(const char* pfx, const char* det, const char* typ, const char* tag );

};
