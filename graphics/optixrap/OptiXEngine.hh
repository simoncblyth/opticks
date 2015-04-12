#pragma once

class Composition ; 
class Renderer ; 
class Texture ; 
class RayTraceConfig ; 
class GGeo ; 

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

// TODO: split off non-OpenGL OptiXCore for headless usage and easier testing

class OptiXEngine {
    public:
        OptiXEngine(const char* cmake_target);

        void setSize(unsigned int width, unsigned int height);
        void setComposition(Composition* composition);
        void setGGeo(GGeo* ggeo);

        void init();
        void initRenderer();
        void initContext();
        void initGeometry();

        void preprocess(); 
        void trace(); 

        void cleanUp();
        void fill_PBO();
        void displayFrame(unsigned int texID);

    public:
        optix::Context& getContext();
        void render();
        
    private: 
        unsigned int getPBOId(){ return m_pbo ; }
        unsigned int getVBOId(){ return m_vbo ; }
        void push_PBO_to_Texture(unsigned int texId);
        void associate_PBO_to_Texture(unsigned int texId);

    protected:
        optix::Buffer createOutputBuffer(RTformat format, unsigned int width, unsigned int height);
        optix::Buffer createOutputBuffer_VBO(RTformat format, unsigned int width, unsigned int height);
        optix::Buffer createOutputBuffer_PBO(RTformat format, unsigned int width, unsigned int height);

    protected:
        optix::Context        m_context; 
        optix::Buffer         m_output_buffer ; 
        optix::GeometryGroup  m_geometry_group ;
        optix::Aabb           m_aabb;

        unsigned int m_width ;
        unsigned int m_height ;
        unsigned int m_vbo ;
        unsigned int m_pbo ;

        size_t m_vbo_element_size;
        size_t m_pbo_element_size;

        unsigned char* m_pbo_data ; 

        Composition*     m_composition ; 
        Renderer*        m_renderer ; 
        Texture*         m_texture ; 
        RayTraceConfig*  m_config ; 
        GGeo*            m_ggeo ; 


};



