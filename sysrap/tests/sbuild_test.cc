/**
sbuild_test.cc
================

~/o/sysrap/tests/sbuild_test.sh 
   script built

sbuild_test 
   CMake built

**/

#include <iostream>
#include "sbuild.h"

int main()
{
    printf("sbuild::BUILD_TYPE [%s]\n", sbuild::BUILD_TYPE ); 

    std::cout << sbuild::Desc() ; 

    return 0 ; 
}

