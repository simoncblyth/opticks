#include "Composition.hh"

#include "NPY.hpp"
#include "GLMPrint.hpp"

void test_rotate()
{
    glm::vec3 X(1,0,0);
    glm::vec3 Y(0,1,0);
    glm::vec3 Z(0,0,1);

    float angle = 0.f ; 
    for(unsigned int i=0 ; i < 6 ; i++)
    {
        switch(i)
        {
            case 0:angle = 0.f ; break;   
            case 1:angle = 30.f ; break;   
            case 2:angle = 45.f ; break;   
            case 3:angle = 60.f ; break;   
            case 4:angle = 90.f ; break;   
            case 5:angle = 180.f ; break;   
        }

        float a = angle*M_PI/180. ; 
        printf(" angle %10.4f a %10.4f \n", angle, a );

        glm::mat4 rotX = glm::rotate(glm::mat4(1.0), a, X );
        glm::mat4 rotY = glm::rotate(glm::mat4(1.0), a, Y );
        glm::mat4 rotZ = glm::rotate(glm::mat4(1.0), a, Z );

        glm::mat4 irotX = glm::transpose(rotX);
        glm::mat4 irotY = glm::transpose(rotY);
        glm::mat4 irotZ = glm::transpose(rotZ);

        print(rotX, "rotX"); 
        print(irotX, "irotX"); 

        print(rotY, "rotY"); 
        print(irotY, "irotY"); 

        print(rotZ, "rotZ"); 
        print(irotZ, "irotZ"); 



   }
}



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
   //test_setCenterExtent();
   test_rotate();

    
   return 0 ;
}
