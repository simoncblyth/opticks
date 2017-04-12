
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

    glm::mat4 irit = nglmext::invert_tr( tr );

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

    /*


                   u
                   

            lu            ru

        la     lb     ra      rb


    */
  
    // lu
    nsphere la = make_nsphere(-500.f,0.f,-50.f,100.f); la.label = "la" ; 
    nsphere lb = make_nsphere(-500.f,0.f, 50.f,100.f); lb.label = "lb" ; 
    nunion  lu = make_nunion( &la, &lb );
    la.parent = &lu ; 
    lb.parent = &lu ; 

    // ru
    nsphere ra = make_nsphere( 500.f,0.f,-50.f,100.f); ra.label = "ra" ; 
    nsphere rb = make_nsphere( 500.f,0.f, 50.f,100.f); rb.label = "rb" ; 
    nunion  ru = make_nunion( &ra, &rb );
    ra.parent = &ru ; 
    rb.parent = &ru ; 

    // u 
    nunion u = make_nunion( &lu, &ru );
    lu.parent = &u ; 
    ru.parent = &u ; 
 

    u.transform = &mpx ; 
    ru.transform = &mpy ; 
    rb.transform = &mpz ;     
     

    // setting gtransform on internal nodes does nothing ... need to do that to the leaves
    u.update_gtransforms();

    assert(ra.gtransform);
    assert(rb.gtransform);
    assert(la.gtransform);
    assert(lb.gtransform);

    std::cout << " rb.gt " << *rb.gtransform << std::endl ; 

    std::vector<glm::vec3> centers ; 
    std::vector<glm::vec3> dirs ; 
    u.collect_prim_centers(centers, dirs);

    unsigned ncen = centers.size();
    unsigned ndir = dirs.size();
    assert( ncen == ndir );

    for(unsigned i=0 ; i < ncen ; i++) std::cout << i << " center:" << centers[i] << " dir:" << dirs[i] << std::endl ; 


    u.dump_prim("prim", 1); 

    std::cout << std::endl ; 
    u.dump_prim("prim", 2); 

    // hmm need to do this with rotations, as with translation order doesnt matter 
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_node_transforms();

    return 0 ; 
}



