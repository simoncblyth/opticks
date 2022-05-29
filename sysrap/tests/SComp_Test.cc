// name=SComp_Test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "SComp.h"

int main()
{
    const char* names = "photon,record,rec" ; 

    unsigned mask = SComp::Mask(names); 

    std::cout 
        << " names " << names << std::endl  
        << " SComp::Mask(names) " << SComp::Mask(names) << std::endl 
        << " mask " << mask << std::endl 
        << " SComp::Desc(mask) " << SComp::Desc(mask) << std::endl 
        ; 

    return 0 ; 
}
