#include "GTestShape.hh"

#include "GSolid.hh"
#include "GMesh.hh"
#include "glm/glm.hpp"

int main()
{
      glm::vec4 spec(0.f,0.f,0.f,100.f) ; 

      GSolid* solid = GTestShape::makeSphere(spec);

      solid->Summary();

      GMesh* mesh = solid->getMesh();

      mesh->dump();
}

