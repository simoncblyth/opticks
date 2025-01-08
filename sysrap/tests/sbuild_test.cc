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
#include "ssys.h"

struct sbuild_test
{
    static int main();  
}; 

inline int sbuild_test::main()
{
    const char* TEST = ssys::getenvvar("TEST", "ContextString"); 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 
    if(ALL||0==strcmp(TEST,"Desc"))          std::cout << sbuild::Desc() << "\n" ; 
    if(ALL||0==strcmp(TEST,"BuildType"))     std::cout << sbuild::BuildType() << "\n" ; 
    if(ALL||0==strcmp(TEST,"RNGName"))       std::cout << sbuild::RNGName() << "\n" ; 
    if(ALL||0==strcmp(TEST,"ContextString")) std::cout << sbuild::ContextString() << "\n" ; 
    return 0 ; 
}

int main(){ return sbuild_test::main() ; }

