#pragma once

#include <functional>
#include <vector>
#include <string>

#include "NGLM.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class Timer ; 
class NTrianglesNPY ; 
class ImplicitMesherF ;
struct nnode ; 
struct nbbox ; 

class NPY_API NImplicitMesher
{
    public:
        typedef std::function<float(float,float,float)> FUNC ; 
    public:
        NImplicitMesher(nnode* node, int resolution=100, int verbosity=1, float scale_bb=1.01f, int ctrl=0, std::string seedstr="");
        NTrianglesNPY* operator()();
 
        void setFunc(FUNC sdf);

        NTrianglesNPY* sphere_test(); 
        std::string desc();
        void profile(const char* s);
        void report(const char* msg="NImplicitMesher::report");
    
    private:
        void init();
        int addSeeds();
        int addManualSeeds();
        int addCenterSeeds();
        NTrianglesNPY* collectTriangles(const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>& norms, const std::vector<glm::ivec3>& tris );

    private:
        Timer*           m_timer ; 
        nnode*           m_node ; 
        nbbox*           m_bbox ; 
        FUNC             m_sdf ;  
        ImplicitMesherF* m_mesher ; 
        int              m_resolution; 
        int              m_verbosity ; 
        float            m_scale_bb ;  
        int              m_ctrl ;  
        std::string      m_seedstr ; 

};
