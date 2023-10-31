// ./sframe_test.sh

#include "sframe.h"

void test_uninitialized()
{
    sframe fr = {} ; 
    std::cout << "fr" << std::endl << fr << std::endl ; 

    std::cout << "fr.m2w.q3.i.w " << fr.m2w.q3.i.w  << std::endl; 
    std::cout << "fr.m2w.q3.f.w " << fr.m2w.q3.f.w  << std::endl; 
}


void test_save_load()
{
    sframe a ; 
    a.frs = "a test of persisting via metadata" ;
    a.ce.x = 1.f ; 
    a.ce.y = 2.f ; 
    a.ce.z = 3.f ; 
    a.ce.w = 4.f ; 
    std::cout << "a" << std::endl << a << std::endl ; 
    a.save("$FOLD"); 

    sframe b = sframe::Load("$FOLD");  
    std::cout << "b" << std::endl << b << std::endl ; 
    assert( strcmp(a.frs, b.frs) == 0 ); 
}

void test_load()
{
    sframe b = sframe::Load("$FOLD");  
    std::cout << "b" << std::endl << b << std::endl ; 
}

void test_setTranslate()
{
    sframe fr ; 
    fr.setTranslate(100.f, 200.f, 300.f) ; 
    fr.prepare(); 

    std::cout << fr << std::endl ; 
    fr.save("$FOLD"); 
}


int main(int argc, char** argv)
{
    test_uninitialized(); 
    /*
    test_save_load() ; 
    test_load(); 
    test_setTranslate(); 
    */


    return 0 ; 
}
