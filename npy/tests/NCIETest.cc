#include "NCIE.hpp"
#include "PLOG.hh"



void test_funcs_0()
{
    float wl = 500.f ; 

    LOG(info) 
           << " wavelength " << wl
           << " X " << cie_X(wl) 
           << " Y " << cie_Y(wl) 
           << " Z " << cie_Z(wl)
           ; 
}


void test_funcs_1()
{
    float wl = 500.f ; 

    LOG(info) 
           << " wavelength " << wl
           << " X " << NCIE::X(wl) 
           << " Y " << NCIE::Y(wl) 
           << " Z " << NCIE::Z(wl)
           ; 
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_funcs_0();
    test_funcs_1();


    return 0 ; 
}
