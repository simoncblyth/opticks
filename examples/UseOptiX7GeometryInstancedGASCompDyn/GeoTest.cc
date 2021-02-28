// name=GeoTest ; glm- ; gcc -g $name.cc Geo.cc Sys.cc Util.cc Shape.cc Grid.cc -lstdc++ -std=c++11 -I. -I$(glm-prefix) -o /tmp/$name && lldb_ /tmp/$name

#include <iostream>
#include "Geo.h"

int main(int argc, char** argv)
{
    Geo geo ; 
    std::cout << geo.desc() << std::endl ; 
    return 0 ; 
}

