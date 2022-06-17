// name=sdomain_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "sdomain.h"

int main()
{
    std::cout << " sdomain::DomainLength() " << sdomain::DomainLength() << std::endl ;  

    sdomain dom ; 
    std::cout << dom.desc() << std::endl ; 

    return 0 ; 
}

