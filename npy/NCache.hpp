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
#include <boost/filesystem.hpp>


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

// TODO: adopt BFile::FormPath

class NPY_API NCache {
   public:
      NCache(const char* dir);
   public:
      std::string path(const char* relative);
      std::string path(const char* tmpl, const char* incl);
      std::string path(const char* tmpl, unsigned int incl);
   private:
      boost::filesystem::path m_cache ; 

};

#include "NPY_TAIL.hh"

