#include "NGLM.hpp"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "PLOG.hh"

void test_decompose()
{
    LOG(info) << "test_decompose" ; 

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

    std::cout << gpresent(" mtr ", mtr)  << std::endl ; 
    std::cout << gpresent(" mtr_r ", mtr_r)  << std::endl ; 
    std::cout << gpresent(" mtr_t ", mtr_t)  << std::endl ; 

    glm::mat3 mrt_r(mrt) ; 
    glm::vec3 mrt_t(mrt[3]);

    std::cout << gpresent(" mrt ", mrt)  << std::endl ; 
    std::cout << gpresent(" mrt_r ", mrt_r)  << std::endl ; 
    std::cout << gpresent(" mrt_t ", mrt_t)  << std::endl ; 

}



void test_decompose_invert()
{
    LOG(info) << "test_decompose_invert" ;
 
    glm::vec3 axis(1,1,1);
    glm::vec3 tlat(0,0,100) ; 
    float angle = 45.f ; 

    glm::mat4 tr(1.f) ;
    tr = glm::translate(tr, tlat );
    tr = glm::rotate(tr, angle, axis );

    /*
    tr follows familar fourth column translation 4x4 matrix layout :
    which is a rotation followed by the translation on multiplying to the right, hence TR

          tr

          ir it t r 

    */

    std::cout << gpresent(" tr ", tr) << std::endl ; 
 
    // dis-member tr into r and t by inspection and separately 
    // transpose the rotation and negate the translation
    glm::mat4 ir = glm::transpose(glm::mat4(glm::mat3(tr)));
    glm::mat4 it = glm::translate(glm::mat4(1.f), -glm::vec3(tr[3])) ; 
    glm::mat4 irit = ir*it ;    // <--- inverse of tr 

    std::cout << gpresent(" ir ", ir) << std::endl ; 
    std::cout << gpresent(" it ", it) << std::endl ; 
    std::cout << gpresent(" irit ", irit ) << std::endl ; 
    std::cout << gpresent(" irit*tr ", irit*tr ) << std::endl ; 
    std::cout << gpresent(" tr*irit ", tr*irit ) << std::endl ; 
}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_decompose();
    test_decompose_invert();

    return 0 ; 
}


