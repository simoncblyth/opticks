#pragma once

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

class NTrianglesNPY ; 

// its tedious having to template every thing to march against...
// perhaps can adjust to use function pointer 

//template <class signed_distance_function>

typedef double (*SDFPtr)(double, double, double) ;

class NPY_API NMarchingCubesNPY {
    public:
         NMarchingCubesNPY();
         NTrianglesNPY* march(SDFPtr sdf, const glm::uvec3& param, const glm::vec3& low, const glm::vec3& high ); 

};
