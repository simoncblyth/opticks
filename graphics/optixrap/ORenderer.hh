#pragma once

#include <cstddef>

class Renderer ; 
class Texture ; 
class OFrame ; 

class ORenderer {
    public:
        ORenderer( OFrame* frame, const char* dir, const char* incl_path);
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

inline ORenderer::ORenderer(OFrame* frame, const char* dir, const char* incl_path)
    :
    m_frame(frame),
    m_renderer(NULL),
    m_texture(NULL),
    m_texture_id(-1),

    m_render_count(0),
    m_render_prep(0),
    m_render_time(0)
{
    init(dir, incl_path);
}


