/**
UseGLMQuat.cc
==============

https://web.engr.oregonstate.edu/~mjb/vulkan/Handouts/quaternions.1pp.pdf

::

   ~/o/examples/UseGLMQuat/UseGLMQuat.sh 

**/


#include <cassert>
#include <iostream>
#include <sstream>
#include <glm/glm.hpp>

/*
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"
*/

#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"

#include <glm/gtx/string_cast.hpp>


void test_combine_rotations()
{
    glm::quat rot1 = glm::angleAxis( glm::radians(45.f), glm::vec3( 0.707f, 0.707f, 0.f ) );
    glm::quat rot2 = glm::angleAxis( glm::radians(90.f), glm::vec3( 1.f , 0.f , 0.f) );

    std::cout << "rot1:" << glm::to_string( rot1 ) << std::endl ; 
    std::cout << "rot2:" << glm::to_string( rot2 ) << std::endl ; 

    glm::quat rot12 = rot2 * rot1;
    std::cout << "rot12:" << glm::to_string( rot12 ) << std::endl ; 

    glm::vec4 v = glm::vec4( 1., 1., 1., 1. );

    glm::vec4 rot12_v = rot12 * v;

    std::cout << "rot12_v:" << glm::to_string( rot12_v ) << std::endl ; 
    
    glm::mat4 rot12Mat = glm::toMat4( rot12 );

    std::cout << "rot12Mat:" << glm::to_string( rot12Mat ) << std::endl ; 

    glm::vec4 rot12Mat_v = rot12Mat * v; 

    std::cout << "rot12Mat_v:" << glm::to_string( rot12Mat_v ) << std::endl ; 
}


int main()
{
    test_combine_rotations(); 

    return 0 ; 
}
