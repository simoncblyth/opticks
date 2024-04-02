// name=U4VersionTest ; g4- ; gcc $name.cc -std=c++11 -I.. -I$(g4-prefix)/include/Geant4 -lstdc++ -o /tmp/$name && /tmp/$name
#include "U4Version.h"
#include <iostream>

int main()
{
    std::cout << U4Version::Desc() << std::endl ; 
    return 0 ; 
}
