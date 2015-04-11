#pragma once

#include <optixu/optixpp_namespace.h>

class OptiXEngine {
    public:
        OptiXEngine();

        void setSize(unsigned int width, unsigned int height);
        void initContext(unsigned int width, unsigned int height);

        void cleanUp();
        void preprocess(); 
        void trace(); 
        void fill_PBO();

    public:
        optix::Context& getContext();
        unsigned int getPBOId(){ return m_pbo ; }
        unsigned int getVBOId(){ return m_vbo ; }
        void associate_PBO_to_Texture(unsigned int texId);

    protected:
        optix::Buffer createOutputBuffer(RTformat format, unsigned int width, unsigned int height);
        optix::Buffer createOutputBuffer_VBO(RTformat format, unsigned int width, unsigned int height);
        optix::Buffer createOutputBuffer_PBO(RTformat format, unsigned int width, unsigned int height);

    protected:
        optix::Context m_context; 

        unsigned int m_width ;
        unsigned int m_height ;
        unsigned int m_vbo ;
        unsigned int m_pbo ;

        size_t m_vbo_element_size;
        size_t m_pbo_element_size;


};



