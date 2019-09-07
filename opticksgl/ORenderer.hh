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

/**
ORenderer
==========

Used from :doc:`OKGLTracer`

**/



class Renderer ; 
class Texture ; 
class OFrame ; 

#include "OKGL_API_EXPORT.hh"
class OKGL_API ORenderer {
    public:
        ORenderer( Renderer* renderer, OFrame* frame, const char* dir, const char* incl_path);
        void render();
        void report(const char* msg="ORenderer::report");
        void setSize(unsigned int width, unsigned int height);
    private:
        void init(const char* dir, const char* incl_path);

    private:
        OFrame*          m_frame ;  
        Renderer*        m_renderer ; 

        Texture*         m_texture ; 
        int              m_texture_id ; 

        unsigned int     m_render_count ; 
        double           m_render_prep ; 
        double           m_render_time ; 
};


