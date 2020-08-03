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
#include <map>

class Shdr ;

#include "OGLRAP_API_EXPORT.hh"
#include "OGLRAP_HEAD.hh"

/**
Prog
------

Canonical instance is m_shader member of RendererBase


**/

class OGLRAP_API Prog {
//      static const char* LOG_NO_FRAGMENT_SHADER ; 

        friend class InstLODCull ; 
   public:
      Prog(const char* basedir, const char* tag, const char* incl_path=NULL, bool ubo=false );

      void setNoFrag(bool nofrag);
      void setVerbosity(unsigned verbosity);
      void createAndLink();

      void createOnly();
      void linkAndValidate();


      std::string desc() const ; 

      void Summary(const char* msg);
      void Print(const char* msg);
      GLuint getId();

      // required attributes/uniforms cause exits when not found
      // non-required return -1 when not found just like 
      // glGetUniformLocation glGetAttribLocation when no such active U or A 
      GLint attribute(const char* name, bool required=true);
      GLint uniform(const char* name, bool required=true);

   private:
      enum Obj_t { Uniform=0, Attribute=1 } ;

      void setup();
      void readSources(const char* tagdir);
      void setInclPath(const char* path); // colon delimited list of directories to look for glsl inclusions

      void traverseLocation(Obj_t obj, GLenum type,  const char* name, bool print);
      void traverseActive(Obj_t obj, bool print);
      void printStatus();

      static const char* GL_type_to_string(GLenum type); 

   private:
      void create();
      void link();
      void validate();
      void collectLocations();
      void dumpUniforms();
      void dumpAttributes();
      void _print_program_info_log();

   private:
      bool                     m_live ;
      char*                    m_tagdir ;
      const char*              m_tag  ;
      bool                     m_ubo ; 
      GLuint                   m_id ;
      std::vector<std::string> m_names;
      std::vector<GLenum>      m_codes;
      std::vector<Shdr*>       m_shaders ;

      typedef std::map<std::string, GLuint> ObjMap_t ; 
      typedef std::map<std::string, GLenum> TypMap_t ; 

      ObjMap_t  m_attributes ;
      ObjMap_t  m_uniforms   ;
      TypMap_t  m_atype ;
      TypMap_t  m_utype ;

      std::string              m_incl_path ; 
      unsigned   m_verbosity ; 
      bool       m_nofrag ; 

};

#include "OGLRAP_TAIL.hh"

