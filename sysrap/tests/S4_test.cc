// name=S4_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include <vector>
#include "S4.h"

struct Demo
{
    std::string name ; 
    const std::string& GetName() const { return name ; }
}; 


int main(int argc, char** argv)
{
    std::vector<Demo> dd ; 
    dd.push_back( Demo {"Red"} );  
    dd.push_back( Demo {"Green"} );  
    dd.push_back( Demo {"Blue"} );  
    for(unsigned i=0 ; i < dd.size() ; i++) 
    {
        std::cout 
            << " d.GetName " << dd[i].GetName() 
            << " S4::Name(d) " << S4::Name<Demo>(&dd[i])  
            << std::endl
            ; 
    }


    return 0 ; 
}
