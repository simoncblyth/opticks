#pragma once

#include <functional>
#include <vector>
#include <string>

#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class Timer ; 
class NTrianglesNPY ; 
struct nnode ; 
struct nbbox ; 

class NPY_API NHybridMesher
{
    public:
        typedef std::function<float(float,float,float)> FUNC ; 
     public:
        NHybridMesher(nnode* node, int level=5, int verbosity=1);
        NTrianglesNPY* operator()();

        std::string desc();
    private:
        Timer*           m_timer ; 
        nnode*           m_node ; 
        nbbox*           m_bbox ; 
        FUNC             m_sdf ;  
        int              m_level ; 
        int              m_nu ; 
        int              m_nv ; 
        int              m_verbosity ; 

};
