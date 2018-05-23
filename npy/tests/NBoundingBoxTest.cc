#include "NBoundingBox.hpp"
#include <iostream>

int main(int, char** argv)
{
    NBoundingBox bb ; 

    glm::vec3 al(-10,-10,-10); 
    glm::vec3 ah(10,10,10) ; 

    glm::vec3 bl(-20,-20,-20); 
    glm::vec3 bh(0,5,10) ; 

    bb.update( al, ah );
    bb.update( bl, bh );

    std::cout << argv[0]
              << " : "
              << bb.description()
              << std::endl 
              ;


    return 0 ; 
}
