// name=stag_test ; gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <cassert>
#include <iostream>
#include "stag.h"

int main()
{
    stagr r = {} ; 

    std::cout << stagc::Desc() << std::endl ; 

    for(int i=0 ; i < 24 ; i++ )  
    {
        float u = float(i)/float(24) ; 
        r.add(i, u); 
        std::cout << r.desc() << std::endl ; 
    }

    std::cout << " sizeof(stag)                           " << sizeof(stag) << std::endl ; 
    std::cout << " sizeof(r.tag)                          " << sizeof(r.tag) << std::endl ; 
    std::cout << " sizeof(unsigned long long)*stag::NSEQ  " << sizeof(unsigned long long)*stag::NSEQ << std::endl << std::endl ; 
    assert( sizeof(stag) == sizeof(unsigned long long)*stag::NSEQ ); 

    std::cout << " sizeof(r.flat)             " << sizeof(r.flat) << std::endl ; 
    std::cout << " sizeof(float)*stag::SLOTS  " << sizeof(float)*stag::SLOTS << std::endl ; 
    assert( sizeof(r.flat) == sizeof(float)*stag::SLOTS ); 


    return 0 ; 
}
