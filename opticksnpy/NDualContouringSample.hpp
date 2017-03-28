#pragma once

#include <string>
#include "NBBox.hpp"
#include "NGLM.hpp"

#include "NPY_API_EXPORT.hh"

class Timer ; 
class NTrianglesNPY ; 
struct nnode ; 

class NPY_API NDualContouringSample 
{
    public:
        NDualContouringSample(int level=5, float threshold=0.1f, float scale_bb=1.01f );
        NTrianglesNPY* operator()(nnode* node); 
        std::string desc();
        void profile(const char* s);
        void report(const char* msg="NDualContouringSample::report");
    private:
        Timer* m_timer ; 
        int    m_level; 
        int    m_octreeSize ; 
        float  m_threshold ; 
        float  m_scale_bb ; 

        glm::ivec3  m_ilow ; 

};
