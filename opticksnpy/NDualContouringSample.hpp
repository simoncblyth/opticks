#pragma once

#include "NBBox.hpp"
#include "NPY_API_EXPORT.hh"

class NTrianglesNPY ; 
struct nnode ; 


class NPY_API NDualContouringSample 
{
    public:
        NDualContouringSample(unsigned log2size=5, float threshold=0.1f, float scale_bb=1.01f );
        NTrianglesNPY* operator()(nnode* node); 
    private:
        int   m_octreeSize ; 
        float m_threshold ; 
        float m_scale_bb ; 
        nbbox m_node_bb ; 


};
