// name=stag_test ; gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "stag.h"

int main()
{
    stag t = {} ; 
    unsigned stag_slot = 0 ; 

    std::cout << stag::Desc() << std::endl ; 

    for(int i=0 ; i < 24 ; i++ )  
    {
        t.add(stag_slot, i); 
        std::cout << t.desc() << std::endl ; 
    }

    return 0 ; 
}
