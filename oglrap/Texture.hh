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

// env/pycuda/pycuda_pyopengl_interop/pixel_buffer.py


// hmm Texture is too general, QuadTexture better

#include "GMesh.hh"
#include "OGLRAP_API_EXPORT.hh"


struct OGLRAP_API Tex {
   Tex(bool zbuf_) : width(0), height(0), rgb(NULL), rgba(NULL), depth(NULL), zbuf(zbuf_) {}

   int width  ;
   int height ;
   unsigned char* rgb ;
   unsigned char* rgba ;
   unsigned char* depth ;
   bool zbuf ; 
};


class OGLRAP_API Texture : public GMesh {
   public:
       Texture(bool zbuf=false);

       void loadPPM(const char* path);  // does not need OpenGL context
       void create();
       void cleanup();
   public:
       void setSize(unsigned int width, unsigned int height);
       unsigned int getId();
       unsigned int getWidth();
       unsigned int getHeight();
   private:
      static const float pvertex[] ;
      static const float pnormal[] ;
      static const float pcolor[] ;
      static const unsigned int pindex[] ;
      static const float ptexcoord[] ;
   private:
       unsigned int m_width ; 
       unsigned int m_height ; 
       unsigned int m_texture_id ; 
       Tex          m_tex ; 
};


