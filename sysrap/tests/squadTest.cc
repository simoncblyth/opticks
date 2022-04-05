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
    */

    test_quad2_eprd(); 


    return 0 ; 
}
