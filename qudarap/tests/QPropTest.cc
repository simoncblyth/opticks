#include "SPropMockup.h"
#include "QPropTest.h"

int main(int argc, char** argv)
{
    const NP* propcom = SPropMockup::CombinationDemo();
    if(propcom == nullptr) std::cerr << "SPropMockup::CombinationDemo() giving null " << std::endl ; 
    if(propcom == nullptr) return 0 ; 

    std::cout << " propcom " << ( propcom ? propcom->sstr() : "-" ) << std::endl ; 

    //int nx = 161 ; 
    int nx = 1601 ; 

    QPropTest<float> qpt(propcom, 0.f, 16.f, nx ) ; 
    qpt.lookup(); 
    qpt.save(); 

    return 0 ; 
}

