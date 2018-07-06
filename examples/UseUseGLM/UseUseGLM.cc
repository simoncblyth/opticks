
#include <iostream>
#include "UseGLM.hh"

#include <glm/gtx/string_cast.hpp>

int main(int argc, char** argv)
{

    float tr = 100.f ; 
    glm::vec2 rot(10,10) ; 

    glm::mat4 pvm = UseGLM::camera( tr, rot ); 


    std::cout << glm::to_string(pvm) << std::endl ; 


    return 0 ; 
}
