#pragma once
#include <cstring>
#include <glm/glm.hpp>

struct gfloat3 ; 
struct guint3 ; 

class GMesh ; 
class GSolid ; 

class GTestSphere {
   public:
       GTestSphere();
   public:
       static GSolid* makeSolid(glm::vec4& spec, unsigned int meshindex=0, unsigned int nodeindex=0);

};


inline GTestSphere::GTestSphere()
{
}

