// name=SLaunchSequence_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "SLaunchSequence.h"

int main(int argc, char** argv)
{
    SLaunchSequence seq(1000000) ; 
    std::cout << seq.desc() << std::endl ; 

    return 0 ; 
}
