
#include "nmat4triple_.hpp"

#include <iostream>
#include "GLMFormat.hpp"
#include "OPTICKS_LOG.hh"

void test_d()
{
    LOG(info) ; 
    double x = 1. ; 
    double y = 2. ; 
    double z = 3. ; 

    const nmat4triple_<double>* t = nmat4triple_<double>::make_translate( x, y, z);
    std::cout << *t << std::endl ; 
}

void test_f()
{
    LOG(info) ; 
    float x = 1.f ; 
    float y = 2.f ; 
    float z = 3.f ; 

    const nmat4triple_<float>* t = nmat4triple_<float>::make_translate( x, y, z);
    std::cout << *t << std::endl ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_d();
    test_f();

    return 0 ; 

}
