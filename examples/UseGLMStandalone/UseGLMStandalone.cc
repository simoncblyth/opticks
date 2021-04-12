#include <cassert>
#include <iostream>
#include <glm/glm.hpp>

int main(int argc, char** argv)
{
#ifdef GLM_HAS_TEMPLATE_ALIASES
    std::cout << "GLM_HAS_TEMPLATE_ALIASES" << std::endl ; 
#else
    std::cout << "OOPS NOT DEFINED : GLM_HAS_TEMPLATE_ALIASES" << std::endl ; 
#endif

    glm::vec4 v(1.f, 2.f, 3.f, 4.f); 

    for(int i=0 ; i < 4 ; i++ ) std::cout << v[i] << " " ; 
    std::cout << std::endl ; 
    assert( v.w == 4.f ); 


    glm::tmat4x4<double> m(1.); 

    for(int i=0 ; i < 4 ; i++ ) for(int j=0 ; j < 4 ; j++) std::cout << m[i][j] << " " ; 
    std::cout << std::endl ; 
    assert( m[3][3] == 1. ); 


    return 0 ; 
}
