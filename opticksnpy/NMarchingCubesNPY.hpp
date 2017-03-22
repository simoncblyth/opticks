#pragma once

#include "NQuad.hpp"
#include "NBBox.hpp"
#include <vector>

#include "NPY_API_EXPORT.hh"

struct nnode ; 
class NTrianglesNPY ; 


class NPY_API NMarchingCubesNPY {
    public:
        NMarchingCubesNPY(int nx, int ny=0, int nz=0);

        //template<typename T> NTrianglesNPY* operator()(T* node); 
        NTrianglesNPY* operator()(nnode* node); 
    private:
         void march(nnode* node);
         NTrianglesNPY* makeTriangles();
    private:
        int m_nx ; 
        int m_ny ; 
        int m_nz ; 

        nbbox  m_node_bb ; 

        double m_isovalue ; 
        double m_scale ; 
        double m_lower[3] ;
        double m_upper[3] ;

        std::vector<double> m_vertices ; 
        std::vector<size_t> m_polygons ; 

        ntrange3<double> m_range ; 



};


