#pragma once

#include <optixu/optixpp_namespace.h>
#include "Touchable.hh"

class OFrame : public Touchable {
    public:
       OFrame(optix::Context& context, unsigned int width, unsigned int height, bool zbuf=false); 
    public:
       void push_PBO_to_Texture(unsigned int texture_id, unsigned int ztexture_id=0);
       void setSize(unsigned int width, unsigned int height);
    public:
       optix::Buffer& getOutputBuffer();
       optix::Buffer& getDepthBuffer();
       unsigned int getWidth();
       unsigned int getHeight();
       bool         hasZBuffer();
    public:
       // fulfil Touchable interface
       unsigned int touch(int ix, int iy);
    private: 
         static void push_Buffer_to_Texture(optix::Buffer& buffer, int buffer_id, int texture_id, bool depth);
    private: 
        void init(unsigned int width, unsigned int height);
        //void associate_PBO_to_Texture(unsigned int texId);
    protected:
        void fill_PBO();

        // create GL buffer VBO/PBO first then address it as OptiX buffer with optix::Context::createBufferFromGLBO  
        // format can be RT_FORMAT_USER 
        optix::Buffer createOutputBuffer_VBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height);
        optix::Buffer createOutputBuffer_PBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height, bool depth=false);
        optix::Buffer createOutputBuffer(RTformat format, unsigned int width, unsigned int height);

   private:
        optix::Context   m_context ;
        optix::Buffer    m_output_buffer ; 
        optix::Buffer    m_touch_buffer ; 
        optix::Buffer    m_zoutput_buffer ; 
        unsigned int     m_pbo ;
        unsigned int     m_zpbo ;
        unsigned char*   m_pbo_data ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 
        unsigned int     m_push_count ; 
        bool             m_zbuf ; 

};


inline OFrame::OFrame(optix::Context& context, unsigned int width, unsigned int height, bool zbuf)
     :
     m_context(context),
     m_pbo(0),
     m_zpbo(0),
     m_pbo_data(NULL),
     m_push_count(0),
     m_zbuf(zbuf)
{
    init(width, height);
}

inline optix::Buffer& OFrame::getOutputBuffer()
{
    return m_output_buffer ; 
}
inline optix::Buffer& OFrame::getDepthBuffer()
{
    return m_zoutput_buffer ; 
}

inline unsigned int OFrame::getWidth()
{
    return m_width ; 
}
inline unsigned int OFrame::getHeight()
{
    return m_height ; 
}
inline bool OFrame::hasZBuffer()
{
    return m_zbuf ; 
}





