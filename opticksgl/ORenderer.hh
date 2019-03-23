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


