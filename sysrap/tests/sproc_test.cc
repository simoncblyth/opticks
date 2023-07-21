// name=sproc_test ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include "sproc.h"

int main()
{
    std::cerr << sproc::ExecutableName() << std::endl ; 
    return 0 ; 
}
