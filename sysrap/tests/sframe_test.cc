// name=sframe_test ; mkdir -p /tmp/$name ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -I$OPTICKS_PREFIX/externals/glm/glm -o /tmp/$name/$name && /tmp/$name/$name 

#include "sframe.h"

const char* FOLD = "/tmp/sframe_test" ; 


int main(int argc, char** argv)
{
    //const char* msg = "a_test_of_persisting_via_metadata" ; 
    const char* msg = "a test of persisting via metadata" ; 

    sframe a ; 

    a.frs = msg ; 

    a.ce.x = 1.f ; 
    a.ce.y = 2.f ; 
    a.ce.z = 3.f ; 
    a.ce.w = 4.f ; 

    std::cout << "a" << std::endl << a << std::endl ; 

    a.save(FOLD); 

    sframe b = sframe::Load(FOLD);  
 
    std::cout << "b" << std::endl << b << std::endl ; 

    assert( strcmp(a.frs, b.frs) == 0 ); 

    return 0 ; 
}