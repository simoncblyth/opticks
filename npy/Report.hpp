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

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API Report {
   public:
      typedef std::vector<std::string> VS ; 
   public:
      static const char* NAME ;  
      static const char* TIMEFORMAT ;  
      static std::string timestamp();
      static std::string name(const char* typ, const char* tag);
      static Report* load(const char* dir);
   public:
      Report();
   public:
      void add(const VS& lines);
      void save(const char* dir, const char* name);
      void load(const char* dir, const char* name);
      void save(const char* dir);
      void dump(const char* msg="Report::dump");

   private:
      VS          m_lines ; 

};

#include "NPY_TAIL.hh"

