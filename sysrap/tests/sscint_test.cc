// name=sscint_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include  -o /tmp/$name && /tmp/$name

#include "scuda.h"
#include "sscint.h"

int main()
{
    //sscint sc ;        // garbage initial values    
    sscint sc = {} ;     // zeroed initial values  

    assert( sizeof(sscint) == sizeof(float)*6*4 ); 

    std::cout << sc.desc() << " sizeof(sscint) " << sizeof(sscint) << std::endl ; 

    return 0 ; 
}
