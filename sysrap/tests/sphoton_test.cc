// name=sphoton_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

#include <iostream>

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"


void test_qphoton()
{
    qphoton qp ; 
    qp.q.zero(); 
    std::cout << qp.q.desc() << std::endl ; 
}

void test_cast()
{
    sphoton p ; 
    quad4& q = (quad4&)p ; 
    q.zero(); 

    p.wavelength = 501.f ; 

    std::cout << q.desc() << std::endl ; 
    std::cout << p.desc() << std::endl ; 
}

void test_ephoton()
{
    sphoton p ; 
    p.ephoton(); 
    std::cout << p.desc() << std::endl ; 
}

void test_sphoton_selector()
{
    sphoton p ; 
    p.ephoton(); 

    unsigned hitmask = 0xdeadbeef ; 
    sphoton_selector s(hitmask) ; 
    assert( s(p) == false ); 

    p.set_flag(hitmask); 
    assert( s(p) == true ); 
}



int main()
{
    test_qphoton(); 
    test_cast(); 
    test_ephoton(); 
    test_sphoton_selector(); 

    return 0 ; 
}
