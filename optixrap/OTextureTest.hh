#pragma once

#include "OXPPNS.hh"

class Opticks ; 
class OContext ; 

#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OTextureTest {
    public:
        OTextureTest(OContext* ocontext, Opticks* opticks); 
    public:
        void launch();
    private:
        void init();
    private:
        OContext*        m_ocontext ; 
        Opticks*         m_opticks ; 
        optix::Context   m_context ;

        int              m_entry_index ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 
 

};


