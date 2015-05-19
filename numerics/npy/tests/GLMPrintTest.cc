#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include <glm/glm.hpp>
#include <string>

int main()
{
   glm::vec3 c = glm::vec3(1.f,2.f,4.f);
   std::string s = gformat(c);
   printf("[%s]\n", s.c_str());

   return 0 ;
}



