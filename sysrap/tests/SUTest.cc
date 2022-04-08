#include <iostream>
#include <vector>
#include "scuda.h"
#include "squad.h"
#include "SU.hh"

void populate( quad4* pp, unsigned num_p, unsigned mask )
{
    for(unsigned i=0 ; i < num_p ; i++)
    {
        quad4& p = pp[i]; 
        p.zero(); 

        p.q0.f.x = float(i*1000) ; 
        p.q3.u.x = i ; 
        p.q3.u.w = i % 3 == 0 ? mask : i  ; 
    }
}

void dump( const quad4* pp, unsigned num_p )
{
    std::cout << " dump num_p:" << num_p << std::endl ; 
    for(unsigned i=0 ; i < num_p ; i++)
    {
        const quad4& h = pp[i]; 
        std::cout 
             << " h " 
             << h.q3.u.x << " "  
             << h.q3.u.y << " "  
             << h.q3.u.z << " "  
             << h.q3.u.w << " "  
             << std::endl 
             ; 
    }
}

void test_monolithic()
{
    std::vector<quad4> pp(10) ;
    unsigned mask = 0xbeefcafe ;
    populate(pp.data(), pp.size(), mask);

    unsigned num_p = pp.size();
    quad4* d_pp = SU::upload(pp.data(), num_p);

    quad4* hit ;
    unsigned num_hit ;
    qselector<quad4> selector(mask);

    SU::select_copy_device_to_host( &hit, num_hit, d_pp, num_p, selector );

    dump( hit, num_hit );
}

void test_presized()
{
    std::vector<quad4> pp(10) ; 
    unsigned mask = 0xbeefcafe ; 
    populate(pp.data(), pp.size(), mask); 

    unsigned num_p = pp.size(); 
    quad4* d_pp = SU::upload(pp.data(), num_p);   

    qselector<quad4> selector(mask); 
    unsigned num_hit = SU::select_count( d_pp, num_p, selector ); 
    std::cout << " num_hit " << num_hit << std::endl ; 

    quad4* hit = new quad4[num_hit] ;
    SU::select_copy_device_to_host_presized( hit, d_pp, num_p, selector, num_hit );

    dump( hit, num_hit );
}

int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 
    test_monolithic();
    //test_presized();
    return 0 ;
}

