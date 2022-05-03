// name=SCAM_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "SCAM.h"

int main()
{
    const char* key = "CAM" ; 
    int cam = SCAM::EValue(key, "orthographic" ); 
    const char * name = SCAM::Name(cam) ; 
    std::cout 
        << key << " " << cam << " name " << name << std::endl ; 
    return 0 ; 
}
