#include "NGLMStream.hpp"

#include "NGenerator.hpp"
#include "NBox.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"


void test_transform_bbox()
{
     nbox a = make_nbox(0.f,0.f,0.f,100.f);      
     nbbox bb = a.bbox();

     glm::vec3 axis(0,0,1);
     glm::vec3 tlate(0,0,100.);
     float angle = 45.f ; 

     glm::mat4 t(1.0f) ; 
     t = glm::rotate(t, angle, axis );
     t = glm::translate(t, tlate );

     std::cout << t << std::endl ; 


     nbbox tbb0 ;
     nbbox::transform_brute( tbb0, bb, t );

     nbbox tbb1 ;
     nbbox::transform( tbb1, bb, t );

     nbbox tbb = bb.transform(t) ; 

     assert( tbb == tbb0 );
     assert( tbb == tbb1 );

     std::cout << " tbb  " << tbb.desc() << std::endl ; 
     std::cout << " tbb0 " << tbb0.desc() << std::endl ; 
     std::cout << " tbb1 " << tbb1.desc() << std::endl ; 
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_transform_bbox();

    return 0 ; 
}



