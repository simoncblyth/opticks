
#include "NGLMStream.hpp"
#include "GLMFormat.hpp"

#include "NNode.hpp"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"


void test_node_transforms()
{
    LOG(info) << "test_node_transforms" ; 

    glm::mat4 m0 ;
    m0 = glm::translate( m0, glm::vec3( 300,0,0 ) );

    glm::mat4 m1 ;
    m1 = glm::translate( m1, glm::vec3(  0,-300,0 ) );

    glm::mat4 m01 = m0 * m1 ;     
    glm::mat4 m10 = m0 * m1 ;     

    std::cout << " m0  " << m0 << std::endl ; 
    std::cout << " m1  " << m1 << std::endl ; 
    std::cout << " m01 " << m01 << std::endl ; 
    std::cout << " m10 " << m10 << std::endl ; 


    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);

    nunion u = make_nunion( &a, &b );

    a.parent = &u ; 
    b.parent = &u ; 

    u.transform = &m0 ; 
    //b.transform = &m1 ;     
     
    b.gtransform = b.global_transform() ;

    assert(b.gtransform);

    std::cout << " b.gt " << *b.gtransform << std::endl ; 

}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_node_transforms();

    return 0 ; 
}



