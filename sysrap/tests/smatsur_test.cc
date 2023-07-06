// name=smatsur_test ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name 

#include "smatsur.h"
#include <iostream>

int main()
{
    std::cout << smatsur::Desc() ; 
    return 0 ; 
}
