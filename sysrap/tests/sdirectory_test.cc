// name=sdirectory_test ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include "sdirectory.h"
#include <iostream>

int main(int argc, char** argv)
{
    const char* path = "/tmp/blyth/red/green/blue/cyan/purple/puce" ; 
    int rc = sdirectory::MakeDirs(path, 0 ); 
    std::cout << " path " << path << " rc " << rc << std::endl ; 
    return rc ; 
}
