#pragma once

#include <string>
#include "NBBox.hpp"
#include "NGLM.hpp"

#include "NPY_API_EXPORT.hh"

class BTimeKeeper ; 
class NTrianglesNPY ; 
struct nnode ; 

class NPY_API NDualContouringSample 
{
    public:
        NDualContouringSample(int nominal=7, int coarse=6, int verbosity=1, float threshold=0.1f, float scale_bb=1.01f );
        NTrianglesNPY* operator()(nnode* node); 
        std::string desc();
        void profile(const char* s);
        void report(const char* msg="NDualContouringSample::report");
    private:
        BTimeKeeper* m_timer ; 
        int    m_nominal; 
        int    m_coarse; 
        int    m_verbosity ; 
        float  m_threshold ; 
        float  m_scale_bb ; 

};
