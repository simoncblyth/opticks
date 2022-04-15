#include <iostream>
#include <vector>
#include "scuda.h"
#include "squad.h"
#include "SU.hh"
#include "OPTICKS_LOG.hh"

unsigned populate( quad4* pp, unsigned num_p, unsigned mask )
{
    unsigned num_hit = 0 ; 
    for(unsigned i=0 ; i < num_p ; i++)
    {
        quad4& p = pp[i]; 
        p.zero(); 

        p.q0.f.x = float(i*1000) ; 
        p.q3.u.x = i ; 

        bool is_hit = i % 3 == 0 ; 
        p.q3.u.w = is_hit ? mask : i  ; 
 
        if(is_hit) num_hit +=1  ; 
    }
    return num_hit ; 
}

void dump( const quad4* pp, unsigned num_p, unsigned mask )
{
    std::cout << " dump num_p:" << num_p << " mask " << std::hex << mask << std::dec << std::endl ; 
    for(unsigned i=0 ; i < num_p ; i++)
    {
        const quad4& h = pp[i]; 
        unsigned flag = h.q3.u.w ; 

        std::cout 
             << " h " 
             << h.q3.u.x << " "  
             << h.q3.u.y << " "  
             << h.q3.u.z << " "  
             << " 0x" << std::hex << flag << std::dec << " "  
             << std::endl 
             ; 

        assert( ( flag & mask ) == mask );  
    }
}

void test_monolithic()
{
    LOG(info); 
    std::vector<quad4> pp(10) ;
    unsigned mask = 0xbeefcafe ;
    unsigned x_num_hit = populate(pp.data(), pp.size(), mask);

    unsigned num_p = pp.size();
    quad4* d_pp = SU::upload(pp.data(), num_p);

    quad4* hit ;
    unsigned num_hit ;
    qselector<quad4> selector(mask);

    SU::deprecated_select_copy_device_to_host( &hit, num_hit, d_pp, num_p, selector );

    assert( x_num_hit == num_hit );  

    dump( hit, num_hit, mask );
}

void test_presized()
{
    LOG(info); 
    std::vector<quad4> pp(10) ; 
    unsigned mask = 0xbeefcafe ; 
    unsigned x_num_hit = populate(pp.data(), pp.size(), mask);

    unsigned num_p = pp.size(); 
    quad4* d_p = SU::upload<quad4>(pp.data(), num_p);   

    qselector<quad4> selector(mask); 
    unsigned num_hit = SU::count_if<quad4>( d_p, num_p, selector ); 
    std::cout 
         << " num_hit " << num_hit 
         << " x_num_hit " << x_num_hit 
         << std::endl
         ; 
    assert( x_num_hit == num_hit ); 
  
    quad4* d_hit = SU::device_alloc<quad4>( num_hit ); 
    SU::copy_if_device_to_device_presized<quad4>( d_hit, d_p, num_p, selector );

    quad4* hit = new quad4[num_hit] ;
    SU::copy_device_to_host_presized<quad4>( hit, d_hit, num_hit ); 

    dump( hit, num_hit, mask  );
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    std::cout << argv[0] << std::endl ; 
    //test_monolithic();
    test_presized();
    return 0 ;
}

