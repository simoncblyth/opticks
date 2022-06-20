// name=stag_test ; gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o /tmp/$name && lldb__ /tmp/$name

#include <iostream>
#include "stag.h"

int main()
{
    stag t = {} ; 

    std::cout << stag::Desc() << std::endl ; 

    for(int i=0 ; i < 24 ; i++ )  
    {
        t.add(i); 
        std::cout << t.desc() << std::endl ; 
    }

    return 0 ; 
}
