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
#include "plog/Severity.h"

class Prog ;

/**
RendererBase
==============

Base class of both geometry(Renderer) and event(Rdr) renderers
that handles the mechanics of compiling and linking of the shader 
source code.

NB ShaderBase would be a better name, in light of transform feedback 
(and maybe compute shaders in future).

**/

#include "OGLRAP_API_EXPORT.hh"
class OGLRAP_API RendererBase {
   public:
      static const plog::Severity LEVEL ; 
   public:
      RendererBase(const char* tag, const char* dir=NULL, const char* incl_path=NULL, bool ubo=false);

      const char* getName() const ; 
      unsigned    getIndex() const ; 
      const char* getShaderTag() const ; 
      const char* getShaderDir() const ; 
      const char* getInclPath() const ; 

      void setIndexBBox(unsigned index, bool bbox=false);
      void setVerbosity(unsigned verbosity);
      void setNoFrag(bool nofrag);
  public:
      void make_shader();   
  public: 
      // split the make, for transform feedback where need to setup varyings between create and link 
      void create_shader();   
      void link_shader();   
  protected:
      Prog*     m_shader ;
      int       m_program ;
      unsigned  m_verbosity ; 
   private:
      const char* m_shaderdir ; 
      const char* m_shadertag ; 
      const char* m_incl_path ; 
      unsigned    m_index ; 
      bool        m_bbox ; 
      const char* m_name ; 

};

