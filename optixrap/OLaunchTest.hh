#pragma once

#include "OXPPNS.hh"
#include <string>
class Opticks ; 
class OContext ; 

#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OLaunchTest {
    public:
        OLaunchTest(OContext* ocontext, Opticks* opticks, const char* ptx="textureTest.cu.ptx", const char* prog="textureTest", const char* exception="exception"); 
        std::string brief();
    public:
        void setWidth(unsigned int width);
        void setHeight(unsigned int height);
        void prelaunch();
        void launch();
        void launch(unsigned width, unsigned height);
    private:
        void init();
    private:
        OContext*        m_ocontext ; 
        Opticks*         m_opticks ; 
        optix::Context   m_context ;

        const char*      m_ptx ; 
        const char*      m_prog ; 
        const char*      m_exception ; 

        int              m_entry_index ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 
 

};


