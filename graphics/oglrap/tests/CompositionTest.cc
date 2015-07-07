#include "Composition.hh"

#include "NPY.hpp"
#include "GLMPrint.hpp"


int main()
{
   NPY<float>* dom = NPY<float>::load("domain", "1");
   dom->dump();
   glm::vec4 ce = dom->getQuad(0,0);
   print(ce, "ce");

   Composition c ; 
   c.setCenterExtent(ce);
   c.update();
   c.dumpAxisData();

   return 0 ;
}
