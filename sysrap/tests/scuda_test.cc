/**

./scuda_test.sh 


               evt 0x7f6413c02000
//qsim.propagate idx 1 bnc 0 cosTheta     1.0000 dir (   -0.2166    -0.9745     0.0578) nrm (   -0.2166    -0.9745     0.0578) 
//qsim.propagate idx 1 bounce 0 command 3 flag 0 s.optical.x 0 
//qsim.propagate_at_boundary idx 1 nrm   (    0.2166     0.9745    -0.0578) 
//qsim.propagate_at_boundary idx 1 mom_0 (   -0.2166    -0.9745     0.0578) 
//qsim.propagate_at_boundary idx 1 pol_0 (   -0.2578     0.0000    -0.9662) 
//qsim.propagate_at_boundary idx 1 c1     1.0000 normal_incidence 0 
//qsim.propagate_at_boundary idx 1 normal_incidence 0 p.pol (   -0.2578,    0.0000,   -0.9662) p.mom (   -0.2166,   -0.9745,    0.0578) o_normal (    0.2166,    0.9745,   -0.0578)
//qsim.propagate_at_boundary idx 1 TransCoeff        nan n1c1     1.3530 n2c2     1.0003 E2_t (       nan,       nan) A_trans (       nan,       nan,       nan) 
//qsim.propagate_at_boundary idx 1 reflect 0 tir 0 TransCoeff        nan u_reflect     0.3725 
//qsim.propagate_at_boundary idx 1 mom_1 (   -0.2166    -0.9745     0.0578) 
//qsim.propagate_at_boundary idx 1 pol_1 (       nan        nan        nan) 
//qsim.propagate idx 1 bnc 1 cosTheta     0.9745 dir (   -0.2166    -0.9745     0.0578) nrm (    0.0000    -1.0000     0.0000) 
//qsim.propagate idx 1 bounce 1 command 3 flag 0 s.optical.x 99 
2022-06-15 02:24:06.094 INFO  [429898] [SEvt::save@944] DefaultDir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest

**/



#include "scuda.h"


void test_cross()
{
    float3 mom = make_float3( -0.2166f,-0.9745f, 0.0578f ); 
    float3 oriented_normal = make_float3(  0.2166f,0.9745f, -0.0578f ); 
    float3 trans = cross( mom, oriented_normal );    
    printf("// trans (%10.7f, %10.7f, %10.7f) \n", trans.x, trans.y, trans.z ); 

    float trans_mag2 = dot(trans, trans) ; 
    bool trans_mag2_is_zero = trans_mag2  == 0.f ; 
    printf("// trans_mag2 %10.9f trans_mag2_is_zero %d \n", trans_mag2, trans_mag2_is_zero );  

    float3 A_trans = normalize(trans) ; 
    printf("// A_trans (%10.7f, %10.7f, %10.7f) \n", A_trans.x, A_trans.y, A_trans.z ); 
}




void test_indexed()
{
    float4 v = make_float4( 0.f , 1.f, 2.f, 3.f ); 
    const float* vv = (const float*)&v ; 
    printf("//test_indexed vv[0] %10.4f vv[1] %10.4f vv[2] %10.4f vv[3] %10.4f \n",vv[0], vv[1], vv[2], vv[3] ); 
}


void test_efloat()
{
    float f = scuda::efloat("f",101.102f); 
    printf("//test_efloat f (%10.4f) \n", f ); 
}

void test_efloat3()
{
    float3 v = scuda::efloat3("v3","3,33,333"); 
    printf("//test_efloat3 v3 (%10.4f %10.4f %10.4f) \n", v.x, v.y, v.z ); 
}

void test_efloat4()
{
    float4 v = scuda::efloat4("v4","4,44,444,4444"); 
    printf("//test_efloat4 v4 (%10.4f %10.4f %10.4f %10.4f) \n", v.x, v.y, v.z, v.w ); 
}

void test_efloat3n()
{
    float3 v = scuda::efloat3n("v3n","1,1,1"); 
    printf("//test_efloat3n v3n (%10.4f %10.4f %10.4f) \n", v.x, v.y, v.z ); 
}



void test_serial()
{
    float4 v = make_float4( 0.001f , 1.004f, 2.00008f, 3.0006f ); 
    std::cout << " v " << v << std::endl ; 

    std::cout << " scuda::serialize(v) " <<  scuda::serialize(v) << std::endl ; 

}




int main()
{
    /*
    test_cross(); 
    test_indexed();  

    test_efloat(); 
    test_efloat3(); 
    test_efloat4(); 
    test_efloat3n(); 
    */
    test_serial(); 


    return 0 ; 
}


