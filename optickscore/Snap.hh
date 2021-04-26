#pragma once

#include <vector>
#include "plog/Severity.h"
#include "NGLM.hpp"

class Opticks ; 
class Composition ; 
class SRenderer ; 
struct NSnapConfig ; 
struct SMeta ; 


#include "OKCORE_API_EXPORT.hh"

class OKCORE_API Snap 
{
    private:
        static const plog::Severity LEVEL ;     
    public:
        Snap( Opticks* ok, SRenderer* renderer, NSnapConfig* config)  ;
        int render(); 
    private:
        void render_one(const char* path);
        void render_many();
        void render_many(const std::vector<glm::vec3>& eyes );
        void save() const ;
    private:
        void getMinMaxAvg(double& mn, double& mx, double& av) const ;
    private:
        Opticks*       m_ok ; 
        Composition*   m_composition ; 
        unsigned       m_numsteps ; 
        SRenderer*     m_renderer ; 
        NSnapConfig*   m_config ; 
        const char*    m_outdir ; 
        const char*    m_nameprefix ; 
        SMeta*         m_meta ; 

        std::vector<double>    m_frame_times ; 
        std::vector<glm::vec3> m_eyes ; 

};





