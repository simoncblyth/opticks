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

#include <cstddef>
#include <string>
#include "plog/Severity.h"

#include "BRAP_API_EXPORT.hh"

template <typename A, typename B> class Map ; 

class BRAP_API BEnv {
   public:
      static const plog::Severity LEVEL ; 
      typedef Map<std::string, std::string> MSS ; 
      static BEnv* load(const char* dir, const char* name);
      static BEnv* load(const char* path);
      static void dumpEnvironment(const char* msg="BEnv::dumpEnvironment", const char* prefix="G4,OPTICKS,DAE,IDPATH");
   public:
      BEnv(char** envp=NULL);
      void save(const char* dir, const char* name);
      void save(const char* path);
      void dump(const char* msg="BEnv::dump");
      void setPrefix(const char* prefix);
      void setEnvironment(bool overwrite=true, bool native=true);

   private:
      void init();
      void readEnv();
      void readFile(const char* dir, const char* name);
      void readFile(const char* path);

      std::string nativePath(const char* val);

   private:
      char**      m_envp ;
      const char* m_prefix ;
      const char* m_path ;
      MSS*        m_all ;        
      MSS*        m_selection ;        

};



