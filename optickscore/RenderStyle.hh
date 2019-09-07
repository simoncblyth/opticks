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
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class Composition ; 

/**
RenderStyle : O-key
======================

Canonical m_render_style instance is ctor resident of Composition

**/

class OKCORE_API RenderStyle 
{
    public:
        RenderStyle(Composition* composition);
    public:
        void nextRenderStyle(unsigned modifiers); 
        void command(const char* cmd) ;
        std::string desc() const ; 

        //typedef enum { R_PROJECTIVE, R_RAYTRACED, R_COMPOSITE,  NUM_RENDER_STYLE } RenderStyle_t ;  
        typedef enum { R_PROJECTIVE, R_COMPOSITE,  NUM_RENDER_STYLE, R_RAYTRACED } RenderStyle_t ;  

        // try always using composite raytrace (now that are switching off rasterized) see Interactor::nextRenderStyle

        static const char* R_PROJECTIVE_ ; 
        static const char* R_RAYTRACED_ ; 
        static const char* R_COMPOSITE_ ; 
        static const char* RenderStyleName(RenderStyle_t style);
    public:
        RenderStyle::RenderStyle_t getRenderStyle() const  ;

        void setRenderStyle(int style) ; 

        const char*   getRenderStyleName() const  ;
        void setRaytraceEnabled(bool raytrace_enabled); // set by OKGLTracer
        void applyRenderStyle();

        bool isProjectiveRender() const ;
        bool isRaytracedRender() const ;
        bool isCompositeRender() const ;

    private:
        Composition*    m_composition ; 
        RenderStyle_t   m_render_style ; 
        bool            m_raytrace_enabled ; 
};

#include "OKCORE_TAIL.hh"
 
