#pragma once

#include "NQuad.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class NTrianglesNPY ; 


// TODO: split this up in method, 
//       keep some state so can see what went wrong 
//       perhaps use adaptive approach when first attempt fails to yield tris


class NPY_API NMarchingCubesNPY {
    public:
        NMarchingCubesNPY(int nx, int ny=0, int nz=0);
        template<typename T> NTrianglesNPY* operator()(T* node); 
    private:
        int m_nx ; 
        int m_ny ; 
        int m_nz ; 
        ntrange3<double> m_range ; 

};


