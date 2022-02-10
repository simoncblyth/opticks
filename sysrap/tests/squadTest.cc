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





int main(int argc, char** argv)
{
    test_qvals_float(); 
    test_qvals_int(); 

    return 0 ; 
}
