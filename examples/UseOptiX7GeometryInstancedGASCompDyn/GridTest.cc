// name=GridTest ; mkdir -p /tmp/GridTestWrite/{0,1} ; glm- ; gcc -g $name.cc Grid.cc Util.cc -lstdc++ -std=c++11 -I. -I$(glm-prefix) -o /tmp/$name && lldb_ /tmp/$name

#include <iostream>
#include "Grid.h"

int main(int argc, char** argv)
{
    Grid g0(3); 
    std::cout << g0.desc() << std::endl ;  
    g0.write("/tmp", "GridTestWrite", 0 ); 

    return 0 ; 
}
