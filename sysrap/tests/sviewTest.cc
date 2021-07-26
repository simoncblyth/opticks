// name=sviewTest ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cassert>
#include <iostream>
#include "sview.h"

void test_uint()
{
    unsigned u0 = 101 ; 
    float f0  = sview::uint_as<float>( u0 ); 
    double d0 = sview::uint_as<double>( u0 ); 

    std::cout << "u0 " << u0 << std::endl ;  
    std::cout << "f0 " << f0 << std::endl ;  
    std::cout << "d0 " << d0 << std::endl ;  

    unsigned u1 = sview::uint_from<float>(f0) ; 
    unsigned u2 = sview::uint_from<double>(d0) ;
 
    std::cout << "u1 " << u1 << std::endl ;  
    std::cout << "u2 " << u2 << std::endl ;  
    assert( u0 == u1 ); 
    assert( u0 == u2 ); 
}


void test_int()
{
    int i0 = -101 ; 
    float f0  = sview::int_as<float>( i0 ); 
    double d0 = sview::int_as<double>( i0 ); 

    std::cout << "i0 " << i0 << std::endl ;  
    std::cout << "f0 " << f0 << std::endl ;  
    std::cout << "d0 " << d0 << std::endl ;  

    int i1 = sview::int_from<float>(f0) ; 
    int i2 = sview::int_from<double>(d0) ;
 
    std::cout << "i1 " << i1 << std::endl ;  
    std::cout << "i2 " << i2 << std::endl ;  
    assert( i0 == i1 ); 
    assert( i0 == i2 ); 
}

int main()
{
    test_uint(); 
    test_int(); 
    return 0 ; 
}
