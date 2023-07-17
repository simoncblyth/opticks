#include "SPropMockup.h"
#include "QPropTest.h"

int main()
{
    const NP* propcom = SPropMockup::CombinationDemo();
    std::cout << " propcom " << ( propcom ? propcom->sstr() : "-" ) << std::endl ; 


    //int nx = 161 ; 
    int nx = 1601 ; 

    QPropTest<float> qpt(propcom, 0.f, 16.f, nx ) ; 
    qpt.lookup(); 
    qpt.save(); 


    return 0 ; 
}
