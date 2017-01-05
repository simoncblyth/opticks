
// clang boolean-solid.cc -lstdc++ && ./a.out && rm a.out

#include <iostream>

#define BOOLEAN_SOLID_DEBUG 1 
#include "boolean-solid.h"

int main(int argc, char** argv)
{
    boolean_action<Union>        uat ; 
    boolean_action<Intersection> iat ; 
    boolean_action<Difference>   dat ; 

    std::cout << uat.dumptable("Union") << std::endl ;
    std::cout << iat.dumptable("Intersection") << std::endl ;
    std::cout << dat.dumptable("Difference") << std::endl ;

    return 0 ; 
}
