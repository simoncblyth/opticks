#pragma once

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

class NTrianglesNPY ; 

template <class signed_distance_function>
class NPY_API NMarchingCubesNPY {
    public:
         NMarchingCubesNPY();
         //NTrianglesNPY* march(signed_distance_function sdf);
         NTrianglesNPY* march(signed_distance_function sdf, const glm::uvec3& param, const glm::vec3& low, const glm::vec3& high ); 

};
