#pragma once

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

class NTrianglesNPY ; 

// its tedious having to template every thing to march against...
// perhaps can adjust to use function pointer, but that gets complicated
// as then need pointers to member functions
//
//typedef double (*SDF)(double, double, double) ;

class NPY_API NMarchingCubesNPY {
    public:
         NMarchingCubesNPY();

         template <typename SDF> NTrianglesNPY* march(SDF sdf, const glm::uvec3& param, const glm::vec3& low, const glm::vec3& high ); 

};
