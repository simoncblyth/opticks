#pragma once
#include <cstring>
#include <glm/glm.hpp>

struct gfloat3 ; 
struct guint3 ; 

class GMesh ; 
class GSolid ; 

class GTestSphere {
   public:
       enum { NUM_VERTICES = 24, 
              NUM_FACES = 6*2 } ;
   public:
       GTestSphere();
   public:
       static void tesselate(float radius, gfloat3* vertices, guint3* faces, gfloat3* normals);
       static GMesh*  makeMesh(glm::vec4&  spec, unsigned int meshindex);
       static GSolid* makeSolid(glm::vec4& spec, unsigned int meshindex, unsigned int nodeindex);

};


inline GTestSphere::GTestSphere()
{
}

