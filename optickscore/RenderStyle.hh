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

        typedef enum { R_PROJECTIVE, R_RAYTRACED, R_COMPOSITE,  NUM_RENDER_STYLE } RenderStyle_t ;  

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
 
