#pragma once

#include <optixu/optixpp_namespace.h>

class OptiXEngine {
    public:
        OptiXEngine();
        optix::Context& getContext();
        void initContext(unsigned int width, unsigned int height);
        void cleanUp();

        void preprocess(); 
        void trace(); 

    public:
        void  setUseVBOBuffer( bool onoff ) { m_use_vbo_buffer = onoff; }
        bool  usesVBOBuffer() { return m_use_vbo_buffer; }

    protected:
        optix::Buffer createOutputBuffer(RTformat format, unsigned int width, unsigned int height);

    protected:
        optix::Context m_context; 

        bool   m_use_vbo_buffer;
        bool   m_cpu_rendering_enabled;
        int    m_num_devices;


};



