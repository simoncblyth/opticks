// name=RGTest ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include <cassert>
#include "RG.h"

int main(int argc, char** argv)
{
    for(int32_t i=0 ; i < 10 ; i++ ) 
    {
        const char* name = RG::Name(i); 
        if(name == nullptr) continue ; 

        int32_t type = RG::Type(name); 
        std::cout 
           << " i " << std::setw(3) << i 
           << " RG::Name(i) " << std::setw(10) << name
           << " type " << std::setw(3) << type 
           << std::endl 
           ; 
        assert( type == i ); 
    }
    return 0 ; 
}
