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

#include <map>
#include <string>

/*
*GColorMap* manages the association of named items with colors
*/

// hmm this is just a string string map, nothing special for color .. 
// replacing with npy-/Map<std::string, std::string> in new GPropertyLib approach 


#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GColorMap  {
   public:
       static GColorMap* load(const char* dir, const char* name);
   public:
       GColorMap();
       void dump(const char* msg="GColorMap::dump");

   public:
       void addItemColor(const char* iname, const char* color); 
       const char* getItemColor(const char* iname, const char* missing=NULL);

   private:
       void loadMaps(const char* idpath, const char* name);

   private:
       std::map<std::string, std::string>  m_iname2color ; 

};

#include "GGEO_TAIL.hh"

