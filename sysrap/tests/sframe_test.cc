// name=sframe_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -I$OPTICKS_PREFIX/externals/glm/glm -o /tmp/$name && /tmp/$name 

#include "sframe.h"


int main(int argc, char** argv)
{
    sframe sf ; 

    std::cout << sf << std::endl ; 


    return 0 ; 
}
