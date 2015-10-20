#pragma once

#include <optixu/optixpp_namespace.h>
#include "Touchable.hh"

class OFrame : public Touchable {
    public:
       OFrame(optix::Context& context, unsigned int width, unsigned int height); 
       optix::Buffer& getOutputBuffer();
       void push_PBO_to_Texture(unsigned int texId);

       void setSize(unsigned int width, unsigned int height);
       unsigned int getWidth();
       unsigned int getHeight();
       // fulfil Touchable interface
       unsigned int touch(int ix, int iy);

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
        unsigned int     m_pbo ;
        unsigned char*   m_pbo_data ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 
        unsigned int     m_push_count ; 

};


inline OFrame::OFrame(optix::Context& context, unsigned int width, unsigned int height)
     :
     m_context(context),
     m_pbo(0),
     m_pbo_data(NULL),
     m_push_count(0)
{
    init(width, height);
}

inline optix::Buffer& OFrame::getOutputBuffer()
{
    return m_output_buffer ; 
}
inline unsigned int OFrame::getWidth()
{
    return m_width ; 
}
inline unsigned int OFrame::getHeight()
{
    return m_height ; 
}




