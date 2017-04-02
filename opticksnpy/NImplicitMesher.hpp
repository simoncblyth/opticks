#pragma once

#include <vector>
#include <string>

#include "NGLM.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class Timer ; 
class NTrianglesNPY ; 
struct nnode ; 

class NPY_API NImplicitMesher
{
    public:
        NImplicitMesher(int resolution=100, int verbosity=1, float scale_bb=1.01f, int ctrl=0);
        NTrianglesNPY* operator()(nnode* node); 
        NTrianglesNPY* sphere_test(); 
        std::string desc();
        void profile(const char* s);
        void report(const char* msg="NImplicitMesher::report");
    
    private:
        NTrianglesNPY* collectTriangles(const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>& norms, const std::vector<glm::ivec3>& tris );

    private:
        Timer* m_timer ; 
        int    m_resolution; 
        int    m_verbosity ; 
        float  m_scale_bb ;  
        int    m_ctrl ;  

};
