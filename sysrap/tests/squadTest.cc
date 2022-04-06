// ./squadTest.sh 

#include "scuda.h"
#include "squad.h"

void test_qvals_float()
{
    float  v1 ; 
    float2 v2 ; 
    float3 v3 ; 
    float4 v4 ; 

    qvals( v1, "TMIN", "0.6" ); 
    qvals( v2, "CHK", "100.5   -200.1" ); 
    qvals( v3, "EYE", ".4,.2,.1" ); 
    qvals( v4, "LOOK", ".4,.2,.1,1" ); 

    std::cout << "v1 " << v1 << std::endl ; 
    std::cout << "v2 " << v2 << std::endl ; 
    std::cout << "v3 " << v3 << std::endl ; 
    std::cout << "v4 " << v4 << std::endl ; 
}


void test_qvals_int()
{
    int  v1 ; 
    int2 v2 ; 
    int3 v3 ; 
    int4 v4 ; 

    qvals( v1, "I1", "101" ); 
    qvals( v2, "I2", "101   -202" ); 
    qvals( v3, "I3", "101 202 303" ); 
    qvals( v4, "I4", "101 -202 +303 -404" ); 

    std::cout << "v1 " << v1 << std::endl ; 
    std::cout << "v2 " << v2 << std::endl ; 
    std::cout << "v3 " << v3 << std::endl ; 
    std::cout << "v4 " << v4 << std::endl ; 
}


void test_qvals_float3_x2()
{
    float3 mom ; 
    float3 pol ; 
    qvals(mom, pol, "MOM_POL", "1,0,0,0,1,0" ); 
   
    std::cout << "mom " << mom << std::endl ; 
    std::cout << "pol " << pol << std::endl ; 
}

void test_qvals_float4_x2()
{
    float4 momw ; 
    float4 polw ; 
    qvals(momw, polw, "MOMW_POLW", "1,0,0,1,0,1,0,1" ); 
   
    std::cout << "momw " << momw << std::endl ; 
    std::cout << "polw " << polw << std::endl ; 
}

void test_quad4_ephoton()
{
    quad4 p ; 
    p.ephoton(); 
    std::cout << p.desc() << std::endl ;  
}

void test_qenvint()
{
   int num = qenvint("NUM", "-1"); 
   std::cout << " num " << num << std::endl ; 
}

void test_quad4_normalize_mom_pol()
{
    quad4 p ; 
    p.zero() ;
    p.q1.f = make_float4( 1.f, 1.f, 1.f, 1.f ); 
    p.q2.f = make_float4( 1.f, 1.f, 0.f, 1.f ); 
 
    std::cout << p.desc() << std::endl ;  
    p.normalize_mom_pol(); 
    std::cout << p.desc() << std::endl ;  

}

void test_quad2_eprd()
{
    quad2 prd = quad2::make_eprd(); 
    std::cout << " prd.desc " << prd.desc() << std::endl ;  
}

void test_qvals_float4_vec(bool normalize_)
{
    std::vector<float4> v ; 
    qvals(v, "SQUADTEST_F4V", "0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4.5,4.5,4.5,4.5", normalize_ ); 
    for(unsigned i=0 ; i < v.size() ; i++) std::cout << v[i] << std::endl ; 
}

void test_quad4_set_flags_get_flags()
{
    quad4 p ; 
    p.zero(); 

    unsigned MISSING = ~0u ; 
    unsigned boundary0 = MISSING ; 
    unsigned identity0 = MISSING ; 
    unsigned idx0 = MISSING ; ; 
    unsigned flag0 = MISSING ; ; 
    float orient0 = 0.f ; 

    p.get_flags(boundary0, identity0, idx0, flag0, orient0 ); 

    assert( boundary0 == 0u ); 
    assert( identity0 == 0u ); 
    assert( idx0 == 0u ); 
    assert( flag0 == 0u ); 
    assert( orient0 == 1.f );  

    unsigned boundary1, identity1, idx1, flag1 ;
    float orient1 ; 

    // test maximum values of the fields  
    boundary1 = 0xffffu ; 
    identity1 = 0xffffffffu ; 
    idx1      = 0x7fffffffu ;  // bit 31 used for orient  
    flag1     = 0xffffu ; 
    orient1   = -1.f ; 

    p.set_flags(boundary1, identity1, idx1, flag1, orient1 ); 

    unsigned boundary2, identity2, idx2, flag2  ;
    float orient2 ; 
 
    p.get_flags(boundary2 , identity2 , idx2 , flag2, orient2  );

    std::cout 
        << " idx1 " << std::hex << idx1 
        << " idx2 " << std::hex << idx2
        << std::dec
        << std::endl 
        ; 

    assert( boundary2 == boundary1 ); 
    assert( identity2 == identity1 ); 
    assert( idx2 == idx1 );
    assert( flag2 == flag1 );  
    assert( orient2 == orient1 );  
}

void test_quad4_set_flag_get_flag()
{
    quad4 p ; 
    p.zero(); 

    unsigned flag0[2] ; 
    flag0[0] = 1024 ; 

    p.set_flag( flag0[0] ); 
    p.get_flag( flag0[1] ); 
    assert( flag0[0] == flag0[1] ); 
    assert( p.q3.u.w == 1024 ); 

    unsigned flag1[2] ; 
    flag1[0] = 2048 ; 

    p.set_flag( flag1[0] ); 
    p.get_flag( flag1[1] ); 
    assert( flag1[0] == flag1[1] ); 
    
    assert( p.q3.u.w == (1024 | 2048) ); 
}


int main(int argc, char** argv)
{
    /*
    test_qvals_float(); 
    test_qvals_int(); 
    test_qvals_float3_x2(); 
    test_qvals_float4_x2(); 
    test_qenvint(); 
    test_quad4_normalize_mom_pol(); 
    test_quad4_ephoton(); 
    test_quad2_eprd(); 
    test_qvals_float4_vec(false); 
    test_qvals_float4_vec(true); 
    test_quad4_set_flags_get_flags(); 
    */
    test_quad4_set_flag_get_flag(); 


    return 0 ; 
}
