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

#include <GL/glew.h>
#include <string>
#include <vector>

#include "OGLRAP_API_EXPORT.hh"
#include "OGLRAP_HEAD.hh"

class OGLRAP_API Shdr {
   //friend class Prog ; 
   public:
       static const char* incl_prefix ; 

       // ctor only reads the file, no context needed
       Shdr(const char* path, GLenum type, const char* incl_path=NULL);

    public:
       void createAndCompile();
       void Print(const char* msg);
       GLuint getId();
 
   private:
       void setInclPath(const char* path, char delim=';'); // semicolon delimited list of directories to look for glsl inclusions
       std::string resolve(const char* name);
       void readFile(const char* path);
       void _print_shader_info_log();

   private:
       char*      m_path ; 
       GLenum     m_type ; 
       GLuint     m_id ;

       std::string              m_content;
       std::vector<std::string> m_lines ; 

       std::string              m_incl_path ; 
       std::vector<std::string> m_incl_dirs ; 

};

#include "OGLRAP_TAIL.hh"

