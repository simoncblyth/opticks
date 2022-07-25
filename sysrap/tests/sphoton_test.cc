// ./sphoton_test.sh 

#include <iostream>
#include <array>

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

void test_sphotond()
{
    sphoton  f ; 
    sphotond d ; 

    assert( sizeof(d) == 2*sizeof(f) ); 
    assert( sizeof(f) == 16*sizeof(float) ); 
    assert( sizeof(d) == 16*sizeof(double) ); 
}

void test_sphoton_Get()
{
    float time_0 =  3.f ; 
    float time_1 = 13.f ; 

    NP* a = NP::Make<float>(2, 4, 4); 
    std::array<float, 32> vv = {{
       0.f,  1.f,  2.f,  time_0, 
       4.f,  5.f,  6.f,  7.f, 
       8.f,  9.f, 10.f, 11.f, 
      12.f, 13.f, 14.f, 15.f,

      10.f, 11.f, 12.f, time_1, 
      14.f, 15.f, 16.f, 17.f, 
      18.f, 19.f, 20.f, 21.f, 
      22.f, 23.f, 24.f, 25.f
     }}; 

    memcpy( a->values<float>(), vv.data(), sizeof(float)*vv.size() ); 

    sphoton p0 ; 
    sphoton::Get(p0, a, 0 ); 
    assert( p0.time == time_0 ); 

    sphoton p1 ; 
    sphoton::Get(p1, a, 1 ); 
    assert( p1.time == time_1 ); 
}


void test_sphotond_Get()
{
    double time_0 =  3. ; 
    double time_1 = 13. ; 

    NP* a = NP::Make<double>(2, 4, 4); 
    std::array<double, 32> vv = {{
       0.,  1.,  2.,  time_0, 
       4.,  5.,  6.,  7., 
       8.,  9., 10., 11., 
      12., 13., 14., 15.,

      10., 11., 12., time_1, 
      14., 15., 16., 17., 
      18., 19., 20., 21., 
      22., 23., 24., 25.
     }}; 

    memcpy( a->values<double>(), vv.data(), sizeof(double)*vv.size() ); 

    sphotond p0 ; 
    sphotond::Get(p0, a, 0 ); 
    assert( p0.time == time_0 ); 

    sphotond p1 ; 
    sphotond::Get(p1, a, 1 ); 
    assert( p1.time == time_1 ); 
}




int main()
{
    /*
    test_qphoton(); 
    test_cast(); 
    test_ephoton(); 
    test_sphoton_selector(); 
    test_sphoton_change_flagmask(); 
    test_digest(); 
    test_sphotond(); 
    */

    test_sphoton_Get(); 
    test_sphotond_Get(); 


    return 0 ; 
}
