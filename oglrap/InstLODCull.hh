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


struct RBuf ; 
struct RBuf4 ; 
class Composition ; 

#include "NGLM.hpp"

#include "RendererBase.hh"
#include "OGLRAP_API_EXPORT.hh"

/**
InstLODCull
=============

Provisioned from Scene, used from paired Renderer

Uniform updates via UBO in RContext, controlled up in Scene

Based on demo code developments in 
/Users/blyth/env/graphics/opengl/instcull/LODCullShader.cc

**/

class OGLRAP_API InstLODCull : public RendererBase  
{
       //friend class Renderer ; 
    public:
       static const unsigned QSIZE ;
       static const unsigned INSTANCE_MINIMUM ; 
       enum { LOD_MAX = 4 } ; 
       static const unsigned LOC_InstanceTransform ;

       InstLODCull(const char* tag, const char* dir=NULL, const char* incl_path=NULL);
       virtual ~InstLODCull();

       std::string desc() const ;
       void setupFork(RBuf* src, RBuf4* dst,  RBuf4* dst_devnull);
       void launch();
  private:
       GLuint createForkVertexArray(RBuf* src, RBuf4* dst) ;
       void initShader();
       void applyFork();
       void applyForkStreamQueryWorkaround();
       void pullback();

  private:
       RBuf*        m_src ; 
       RBuf4*       m_dst ; 
       RBuf4*       m_dst_devnull ; 

       unsigned     m_num_instance ; 
       unsigned     m_num_lod ; 
       unsigned     m_launch_count ; 
 
       GLuint       m_lodQuery[LOD_MAX] ; 
       GLuint       m_forkVAO ;  
       GLuint       m_workaroundVAO ;  
};



