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
template<typename T> class NPY ; 

#include "plog/Severity.h"
#include "NPY_API_EXPORT.hh"

class NPY_API ImageNPY {
   public:  
       static const plog::Severity LEVEL ; 
       static NPY<unsigned char>* LoadPPMConcat(const std::vector<std::string>& paths, const bool yflip, const unsigned ncomp, const char* config);
       static NPY<unsigned char>* LoadPPM(const char* path, const bool yflip, const unsigned ncomp, const char* config, bool layer_dimension);
   public:  
       static void SavePPM(const char* path, const NPY<unsigned char>* a, const bool yflip ); 
       static void SavePPM(const char* dir, const char* name,  const NPY<unsigned char>* a, const bool yflip); 
   private:  
       static void SavePPMImp(const char* path,  const NPY<unsigned char>* a, const bool yflip); 

};

