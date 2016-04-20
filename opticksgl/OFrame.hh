#pragma once

#include <optixu/optixpp_namespace.h>

class Texture ; 

#include "Touchable.hh"

class OFrame : public Touchable {
    public:
       OFrame(optix::Context& context, unsigned int width, unsigned int height ); 
    public:
       void push_PBO_to_Texture();
       void setSize(unsigned int width, unsigned int height);
    public:
       optix::Buffer& getOutputBuffer();
       Texture* getTexture();
       int getTextureId();
       unsigned int getWidth();
       unsigned int getHeight();
    public:
       // fulfil Touchable interface
       unsigned int touch(int ix, int iy);
    private: 
         static void push_Buffer_to_Texture(optix::Buffer& buffer, int buffer_id, int texture_id, bool depth=false);
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

        Texture*         m_texture ; 
        int              m_texture_id ; 

        unsigned int     m_pbo ;
        unsigned char*   m_pbo_data ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 
        unsigned int     m_push_count ; 

};


inline OFrame::OFrame(optix::Context& context, unsigned int width, unsigned int height)
     :
     m_context(context),
     m_texture(NULL),
     m_texture_id(-1),
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

inline Texture* OFrame::getTexture()
{
    return m_texture ; 
}

inline int OFrame::getTextureId()
{
    return m_texture_id ; 
}

inline unsigned int OFrame::getWidth()
{
    return m_width ; 
}
inline unsigned int OFrame::getHeight()
{
    return m_height ; 
}





