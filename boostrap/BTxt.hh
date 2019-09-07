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


#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BTxt {
   public:
       typedef std::vector<std::string> VS_t ; 
   public:
       static BTxt* Load(const char* path); 
       BTxt(const char* path = NULL); 
       void read();
   public:
       std::string desc() const ; 
       void dump(const char* msg="BTxt::dump") const ;
       const std::string& getString(unsigned int num) const ; 
       const char* getLine(unsigned int num) const ; 
       unsigned int getNumLines() const ;
       unsigned int getIndex(const char* line) const ; // index of line or UINT_MAX if not found
       void write(const char* path=NULL) const ;
       void prepDir(const char* path=NULL) const ; 
   public:
       void addLine(const std::string& line); 
       void addLine(const char* line); 
       template<typename T> void addValue(T value); 
       const std::vector<std::string>& getLines() const ; 
   private:
       const char* m_path ; 
       VS_t m_lines ; 

};

#include "BRAP_TAIL.hh"

