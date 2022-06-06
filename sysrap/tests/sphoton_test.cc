// name=sphoton_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

#include <iostream>

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "OpticksPhoton.h"


void test_qphoton()
{
    qphoton qp ; 
    qp.q.zero(); 
    std::cout << qp.q.desc() << std::endl ; 
}

void test_cast()
{
    sphoton p ; 
    quad4& q = (quad4&)p ; 
    q.zero(); 

    p.wavelength = 501.f ; 

    std::cout << q.desc() << std::endl ; 
    std::cout << p.desc() << std::endl ; 
}

void test_ephoton()
{
    sphoton p ; 
    p.ephoton(); 
    std::cout << p.desc() << std::endl ; 
}

void test_sphoton_selector()
{
    sphoton p ; 
    p.ephoton(); 

    unsigned hitmask = 0xdeadbeef ; 
    sphoton_selector s(hitmask) ; 
    assert( s(p) == false ); 

    p.set_flag(hitmask); 
    assert( s(p) == true ); 
}


#include <bitset>

void test_sphoton_change_flagmask()
{
    unsigned flagmask = 0u ; 

    flagmask |= CERENKOV ; 
    flagmask |= SCINTILLATION ;
    flagmask |= MISS ; 
    flagmask |= BULK_ABSORB ; 
    flagmask |= BULK_REEMIT ;
    flagmask |= BULK_SCATTER ;
    flagmask |= SURFACE_DETECT ;
    flagmask |= SURFACE_ABSORB ;  
    flagmask |= SURFACE_DREFLECT ; 
    flagmask |= SURFACE_SREFLECT ;
    flagmask |= BOUNDARY_REFLECT  ;
    flagmask |= BOUNDARY_TRANSMIT ;
    flagmask |= TORCH ;
    flagmask |= NAN_ABORT ; 
    flagmask |= EFFICIENCY_CULL ;
    flagmask |= EFFICIENCY_COLLECT ; 

    std::cout << std::setw(20) << " init " << std::bitset<32>(flagmask) << std::endl ; 

    flagmask &= ~BULK_ABSORB ; std::cout << std::setw(20) << " &= ~BULK_ABSORB  " << std::bitset<32>(flagmask) << std::endl ; 
    flagmask &= ~BULK_REEMIT ; std::cout << std::setw(20) << " &= ~BULK_REEMIT  " << std::bitset<32>(flagmask) << std::endl ; 
    flagmask |=  BULK_REEMIT ; std::cout << std::setw(20) << " |= BULK_REEMIT  "  << std::bitset<32>(flagmask) << std::endl ; 
    flagmask |=  BULK_ABSORB ; std::cout << std::setw(20) << " |= BULK_ABSORB  "  << std::bitset<32>(flagmask) << std::endl ; 
}


int main()
{
    /*
    test_qphoton(); 
    test_cast(); 
    test_ephoton(); 
    test_sphoton_selector(); 
    */
    test_sphoton_change_flagmask(); 

    return 0 ; 
}
