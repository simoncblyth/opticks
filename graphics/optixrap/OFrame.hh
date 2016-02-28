#pragma once

#include <optixu/optixpp_namespace.h>
#include "Touchable.hh"

class OFrame : public Touchable {
    public:
       OFrame(optix::Context& context, unsigned int width, unsigned int height, bool zbuffer=false); 
       optix::Buffer& getOutputBuffer();
       optix::Buffer& getDepthBuffer();

       void push_PBO_to_Texture(unsigned int texId);

       void setSize(unsigned int width, unsigned int height);
       unsigned int getWidth();
       unsigned int getHeight();
       bool         hasZBuffer();
       // fulfil Touchable interface
       unsigned int touch(int ix, int iy);

    private: 
         static void push_Buffer_to_Texture(optix::Buffer& buffer, int buffer_id, int texture_id, bool zbuffer);
    private: 
        void init(unsigned int width, unsigned int height);
        void associate_PBO_to_Texture(unsigned int texId);

    protected:
        void fill_PBO();

        // create GL buffer VBO/PBO first then address it as OptiX buffer with optix::Context::createBufferFromGLBO  
        // format can be RT_FORMAT_USER 
        optix::Buffer createOutputBuffer_VBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height);
        optix::Buffer createOutputBuffer_PBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height);
        optix::Buffer createOutputBuffer(RTformat format, unsigned int width, unsigned int height);

   private:
        optix::Context   m_context ;
        optix::Buffer    m_output_buffer ; 
        optix::Buffer    m_touch_buffer ; 
        optix::Buffer    m_depth_buffer ; 
        unsigned int     m_pbo ;
        unsigned int     m_depth ;
        unsigned char*   m_pbo_data ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 
        unsigned int     m_push_count ; 
        bool             m_zbuffer ; 

};


inline OFrame::OFrame(optix::Context& context, unsigned int width, unsigned int height, bool zbuffer)
     :
     m_context(context),
     m_pbo(0),
     m_depth(0),
     m_pbo_data(NULL),
     m_push_count(0),
     m_zbuffer(zbuffer)
{
    init(width, height);
}

inline optix::Buffer& OFrame::getOutputBuffer()
{
    return m_output_buffer ; 
}
inline optix::Buffer& OFrame::getDepthBuffer()
{
    return m_depth_buffer ; 
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
    return m_zbuffer ; 
}





