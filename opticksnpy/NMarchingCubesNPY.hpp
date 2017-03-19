#pragma once

#include "NQuad.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class NTrianglesNPY ; 

class NPY_API NMarchingCubesNPY {
    public:
        NMarchingCubesNPY(const nuvec3& param);

        template<typename T> NTrianglesNPY* operator()(T* node); 
    private:
        nuvec3 m_param ; 

};



// its tedious having to template every thing to march against...
// perhaps can adjust to use function pointer, but that gets complicated
// as then need pointers to member functions
//
//typedef double (*SDF)(double, double, double) ;



