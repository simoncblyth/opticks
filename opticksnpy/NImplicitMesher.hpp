#pragma once

#include <string>
#include "NBBox.hpp"
#include "NGLM.hpp"

#include "NPY_API_EXPORT.hh"

class Timer ; 
class NTrianglesNPY ; 
struct nnode ; 

class NPY_API NImplicitMesher
{
    public:
        NImplicitMesher(int resolution=100, int verbosity=1, float scale_bb=1.01f );
        NTrianglesNPY* operator()(nnode* node); 
        std::string desc();
        void profile(const char* s);
        void report(const char* msg="NImplicitMesher::report");
    private:
        Timer* m_timer ; 
        int    m_resolution; 
        int    m_verbosity ; 
        float  m_scale_bb ; 

};
