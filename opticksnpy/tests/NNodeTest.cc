
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NNode.hpp"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"


nmat4pair make_tlate_pair( const glm::vec3& tlate )
{
    glm::mat4 tr(1.f) ;
    tr = glm::translate( tr, tlate );

    glm::mat4 irit = invert_tr( tr );

    return nmat4pair(tr, irit);
}




void test_node_transforms()
{
    LOG(info) << "test_node_transforms" ; 

    nmat4pair mpx = make_tlate_pair( glm::vec3(300,0,0) );
    nmat4pair mpy = make_tlate_pair( glm::vec3(0,300,0) );
    nmat4pair mpz = make_tlate_pair( glm::vec3(0,0,300) );

    std::cout << " mpx  " << mpx << std::endl ; 
    std::cout << " mpy  " << mpy << std::endl ; 
    std::cout << " mpz  " << mpz << std::endl ; 


    nsphere la = make_nsphere(-500.f,0.f,-50.f,100.f);
    nsphere lb = make_nsphere(-500.f,0.f, 50.f,100.f);
    nunion  lu = make_nunion( &la, &lb );
    la.parent = &lu ; 
    lb.parent = &lu ; 

    nsphere ra = make_nsphere( 500.f,0.f,-50.f,100.f);
    nsphere rb = make_nsphere( 500.f,0.f, 50.f,100.f);
    nunion  ru = make_nunion( &ra, &rb );
    ra.parent = &ru ; 
    rb.parent = &ru ; 

    nunion u = make_nunion( &lu, &ru );
    lu.parent = &u ; 
    ru.parent = &u ; 
 
    u.transform = &mpx ; 
    ru.transform = &mpy ; 
    rb.transform = &mpz ;     
     
    rb.gtransform = rb.global_transform() ;

    assert(rb.gtransform);

    std::cout << " rb.gt " << *rb.gtransform << std::endl ; 


    // hmm need to do this with rotations, as with translation order doesnt matter 

}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_node_transforms();

    return 0 ; 
}



