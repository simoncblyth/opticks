#include "Composition.hh"

#include "NPY.hpp"
#include "GLMPrint.hpp"





void test_center_extent()
{
   NPY<float>* dom = NPY<float>::load("domain", "1", "dayabay");
   dom->dump();
   glm::vec4 ce = dom->getQuad(0,0);
   print(ce, "ce");

   Composition c ; 
   c.setCenterExtent(ce);
   c.update();
   c.dumpAxisData();
}

void test_setCenterExtent()
{
    glm::vec4 ce(100.,100.,100.,500.);
    glm::vec3 sc(ce.w);
    glm::vec3 tr(ce.x, ce.y, ce.z);

    glm::vec3 isc(1.f/ce.w);

    glm::mat4 m_model_to_world = glm::scale( glm::translate(glm::mat4(1.0), tr), sc); 

    glm::mat4 m_world_to_model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr); 
 

    glm::mat4 check = m_world_to_model * m_model_to_world ;

    print(m_model_to_world, "m_model_to_world");
    print(m_world_to_model, "m_world_to_model");
    print(check, "check");
}


int main()
{
   //test_center_extent();
   test_setCenterExtent();

    
   return 0 ;
}
