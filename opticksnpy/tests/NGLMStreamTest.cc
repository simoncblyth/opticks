#include "NGLMStream.hpp"

int main()
{
    glm::ivec3 iv[4] = {
       {1,2,3},
       {10,20,30},
       {100,200,300},
       {1000,2000,3},
    };

    for(int i=0 ; i < 4 ; i++)
        std::cout << std::setw(20) << glm::to_string(iv[i]) << std::endl ; 

    for(int i=0 ; i < 4 ; i++)
        std::cout << std::setw(20) << iv[i] << std::endl ; 



    glm::vec3 fv[4] = {
       {1.23,2.45,3},
       {10.12345,20,30.2235263},
       {100,200,300},
       {1000,2000,3},
    };

    for(int i=0 ; i < 4 ; i++)
        std::cout << std::setw(20) << glm::to_string(fv[i]) << std::endl ; 

    for(int i=0 ; i < 4 ; i++)
        std::cout << std::setw(20) << fv[i] << std::endl ; 






    return 0 ; 
}
