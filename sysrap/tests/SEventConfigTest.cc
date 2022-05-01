#include <iostream>
#include "SEventConfig.hh"

int main(int argc, char** argv)
{
    SEventConfig::SetMaxPhoton(101); 
    std::cout << SEventConfig::Desc() << std::endl ; 


    return 0 ; 
}
