
#include "NGLMStream.hpp"

#include "NNode.hpp"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"


void test_decompose()
{
    glm::vec3 axis(1,1,1);
    glm::vec3 tlat(0,0,100) ; 
    float angle = 45.f ; 

    glm::mat4 mtr ;
    mtr = glm::translate(mtr, tlat );
    mtr = glm::rotate(mtr, angle, axis );
 
    glm::mat4 mrt ;
    mrt = glm::translate(mrt, tlat );
    mrt = glm::rotate(mrt, angle, axis );

    // hmm : the above way of constructing matrix does 
    //       yields the same matrix no matter the order

    assert( mtr == mrt );
 
    glm::mat3 mtr_r(mtr) ; 
    glm::vec3 mtr_t(mtr[3]);

    std::cout << "mtr " << mtr << std::endl ; 
    std::cout << "mtr_r " << mtr_r << std::endl ; 
    std::cout << "mtr_t " << mtr_t << std::endl ; 

    glm::mat3 mrt_r(mrt) ; 
    glm::vec3 mrt_t(mrt[3]);

    std::cout << "mrt " << mrt << std::endl ; 
    std::cout << "mrt_r " << mrt_r << std::endl ; 
    std::cout << "mrt_t " << mrt_t << std::endl ; 
}


void test_decompose_invert()
{
    glm::vec3 axis(1,1,1);
    glm::vec3 tlat(0,0,100) ; 
    float angle = 45.f ; 

    glm::mat4 mtr ;
    mtr = glm::translate(mtr, tlat );
    mtr = glm::rotate(mtr, angle, axis );
 
    glm::mat3 r(mtr) ; 
    glm::vec3 t(mtr[3]);

    glm::mat3 ir = glm::transpose(r);
    glm::vec3 it = -t ; 


    glm::mat4 imtr(1.f);
    imtr[0] = glm::vec4( ir[0], 0.f );
    imtr[1] = glm::vec4( ir[1], 0.f );
    imtr[2] = glm::vec4( ir[2], 0.f );
    imtr[3] = glm::vec4( it ,  1.f );

    glm::mat4 id = imtr * mtr ; 
    glm::mat4 id2 = mtr * imtr ; 

    std::cout << "  mtr " << mtr << std::endl ; 
    std::cout << " imtr " << imtr << std::endl ; 
    std::cout << " id " << id << std::endl ; 
    std::cout << " id2 " << id2 << std::endl ; 



}



void test_node_transforms()
{

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

    //test_node_transforms();
    test_decompose();
    test_decompose_invert();

    return 0 ; 
}



