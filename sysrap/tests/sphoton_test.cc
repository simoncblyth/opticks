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

std::string dump_(const char* label, unsigned mask)
{
    std::bitset<32> bs(mask); 
    std::stringstream ss ; 
    ss << std::setw(25) << label << " " << bs << " " << std::setw(2) << bs.count() ; 
    std::string s = ss.str(); 
    return s ; 
}

void dump( const char* label, unsigned mask) 
{
    std::cout << dump_(label, mask) << std::endl ; 
}

/**

                     zero 00000000000000000000000000000000  0
              |= CERENKOV 00000000000000000000000000000001  1
         |= SCINTILLATION 00000000000000000000000000000011  2
                  |= MISS 00000000000000000000000000000111  3
           |= BULK_ABSORB 00000000000000000000000000001111  4
           |= BULK_REEMIT 00000000000000000000000000011111  5
          |= BULK_SCATTER 00000000000000000000000000111111  6
        |= SURFACE_DETECT 00000000000000000000000001111111  7
        |= SURFACE_ABSORB 00000000000000000000000011111111  8
      |= SURFACE_DREFLECT 00000000000000000000000111111111  9
      |= SURFACE_SREFLECT 00000000000000000000001111111111 10
      |= BOUNDARY_REFLECT 00000000000000000000011111111111 11
     |= BOUNDARY_TRANSMIT 00000000000000000000111111111111 12
                 |= TORCH 00000000000000000001111111111111 13
             |= NAN_ABORT 00000000000000000011111111111111 14
       |= EFFICIENCY_CULL 00000000000000000111111111111111 15
    |= EFFICIENCY_COLLECT 00000000000000001111111111111111 16
          &= ~BULK_ABSORB 00000000000000001111111111110111 15
          &= ~BULK_REEMIT 00000000000000001111111111100111 14
          |=  BULK_REEMIT 00000000000000001111111111110111 15
          |=  BULK_ABSORB 00000000000000001111111111111111 16

**/

void test_sphoton_change_flagmask()
{
    unsigned flagmask = 0u ;         dump("zero", flagmask );  
    flagmask |= CERENKOV ;           dump("|= CERENKOV", flagmask ); 
    flagmask |= SCINTILLATION ;      dump("|= SCINTILLATION", flagmask ); 
    flagmask |= MISS ;               dump("|= MISS", flagmask ); 
    flagmask |= BULK_ABSORB ;        dump("|= BULK_ABSORB", flagmask ); 
    flagmask |= BULK_REEMIT ;        dump("|= BULK_REEMIT", flagmask ); 
    flagmask |= BULK_SCATTER ;       dump("|= BULK_SCATTER", flagmask ); 
    flagmask |= SURFACE_DETECT ;     dump("|= SURFACE_DETECT", flagmask ); 
    flagmask |= SURFACE_ABSORB ;     dump("|= SURFACE_ABSORB", flagmask );   
    flagmask |= SURFACE_DREFLECT ;   dump("|= SURFACE_DREFLECT", flagmask ); 
    flagmask |= SURFACE_SREFLECT ;   dump("|= SURFACE_SREFLECT", flagmask ); 
    flagmask |= BOUNDARY_REFLECT ;   dump("|= BOUNDARY_REFLECT", flagmask ); 
    flagmask |= BOUNDARY_TRANSMIT ;  dump("|= BOUNDARY_TRANSMIT", flagmask ); 
    flagmask |= TORCH ;              dump("|= TORCH", flagmask); 
    flagmask |= NAN_ABORT ;          dump("|= NAN_ABORT", flagmask );  
    flagmask |= EFFICIENCY_CULL ;    dump("|= EFFICIENCY_CULL", flagmask ); 
    flagmask |= EFFICIENCY_COLLECT ; dump("|= EFFICIENCY_COLLECT", flagmask) ;  
    flagmask &= ~BULK_ABSORB ;       dump("&= ~BULK_ABSORB", flagmask); 
    flagmask &= ~BULK_REEMIT ;       dump("&= ~BULK_REEMIT", flagmask); 
    flagmask |=  BULK_REEMIT ;       dump("|=  BULK_REEMIT", flagmask); 
    flagmask |=  BULK_ABSORB ;       dump("|=  BULK_ABSORB", flagmask); 
}

void test_digest()
{
    sphoton p ; 
    p.ephoton(); 

    std::cout 
        << " p.digest()   " << p.digest() << std::endl 
        << " p.digest(16) " << p.digest(16) << std::endl 
        << " p.digest(12) " << p.digest(12) << std::endl 
        ; 

}


int main()
{
    /*
    test_qphoton(); 
    test_cast(); 
    test_ephoton(); 
    test_sphoton_selector(); 
    test_sphoton_change_flagmask(); 
    */

    test_digest(); 

    return 0 ; 
}
