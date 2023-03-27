// ./sframe_test.sh

#include "sframe.h"

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


int main(int argc, char** argv)
{
    /*
    test_save_load() ; 
    */

    test_load(); 

    return 0 ; 
}
