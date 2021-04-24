#pragma once

#include <vector>
#include "plog/Severity.h"
#include "NGLM.hpp"

class Opticks ; 
class Composition ; 
class SRenderer ; 
struct NSnapConfig ; 


#include "OKCORE_API_EXPORT.hh"

class OKCORE_API Snap 
{
    private:
        static const plog::Severity LEVEL ;     
    public:
        Snap( Opticks* ok, SRenderer* renderer, NSnapConfig* config)  ;
        void render(); 

    private:
        void render_one(const char* path);
        void render_many();
        void render_many(const std::vector<glm::vec3>& eyes );
       
    private:
        Opticks*       m_ok ; 
        Composition*   m_composition ; 
        unsigned       m_numsteps ; 
        SRenderer*     m_renderer ; 
        NSnapConfig*   m_config ; 

        std::vector<glm::vec3> m_eyes ; 

};





