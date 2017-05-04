#include "NQuad.hpp"
#include <iostream>
#include <cstdio>



/*
 ///  copy sign-bit from src value to dst value
 183 OPTIXU_INLINE float copysignf(const float dst, const float src)
 184 {
 185   union {
 186     float f;
 187     unsigned int i;
 188   } v1, v2, v3;
 189   v1.f = src;
 190   v2.f = dst;
 191   v3.i = (v2.i & 0x7fffffff) | (v1.i & 0x80000000);
 192 
 193   return v3.f;
 194 }
 195 
*/





void test_negative_zero()
{
    nquad q, p ;
    q.u = {0,1,2,3} ;
    p.u = {0,0,0,0} ;

    const unsigned SIGNBIT32 = 0x1 << 31  ;
    const unsigned OTHERBIT32 = 0x7fffffffu ;

    assert(  SIGNBIT32  == 0x80000000u );
    // assert(  !SIGNBIT32 == OTHERBIT32 );  // <<< not so


    q.i.x = (q.i.x & OTHERBIT32) | SIGNBIT32 ;  // <- set the sign bit 
    q.i.y = (q.i.y & OTHERBIT32) | SIGNBIT32 ;  // <- set the sign bit 
    q.i.z = (q.i.z & OTHERBIT32) | SIGNBIT32 ;  // <- set the sign bit 
    q.i.w = (q.i.w & OTHERBIT32) | SIGNBIT32 ;  // <- set the sign bit 

    p.u.x =   q.u.x & OTHERBIT32 ; 
    p.u.y =   q.u.y & OTHERBIT32 ; 
    p.u.z =   q.u.z & OTHERBIT32 ; 
    p.u.w =   q.u.w & OTHERBIT32 ; 


/*
test_negative_zero q.i.x 0 q.u.x 0 q.f.x 0 q.i.y 0 q.u.y 0 q.f.y 0 q.i.z 0 q.u.z 0 q.f.z 0 q.i.w -2147483648 q.u.w 2147483648 q.f.w -0

test_negative_zero
 q.i.x -2147483648 q.u.x 2147483648 q.f.x -0
 q.i.y -2147483647 q.u.y 2147483649 q.f.y -1.4013e-45
 q.i.z -2147483646 q.u.z 2147483650 q.f.z -2.8026e-45
 q.i.w -2147483645 q.u.w 2147483651 q.f.w -4.2039e-45

*/

    std::cout << "test_negative_zero" << std::endl ; 


    std::cout 
              << " q.i.x " << q.i.x 
              << " q.u.x " << q.u.x 
              << " q.f.x " << q.f.x 
              << " p.i.x " << p.i.x 
              << " p.u.x " << p.u.x 
              << " p.f.x " << p.f.x 
              << std::endl ; 

    std::cout 
              << " q.i.y " << q.i.y 
              << " q.u.y " << q.u.y 
              << " q.f.y " << q.f.y 
              << " p.i.y " << p.i.y 
              << " p.u.y " << p.u.y 
              << " p.f.y " << p.f.y 
              << std::endl ; 

    std::cout 
              << " q.i.z " << q.i.z 
              << " q.u.z " << q.u.z 
              << " q.f.z " << q.f.z 
              << " p.i.z " << p.i.z 
              << " p.u.z " << p.u.z 
              << " p.f.z " << p.f.z 
              << std::endl ; 

    std::cout 
              << " q.i.w " << q.i.w 
              << " q.u.w " << q.u.w 
              << " q.f.w " << q.f.w 
              << " p.i.w " << p.i.w 
              << " p.u.w " << p.u.w 
              << " p.f.w " << p.f.w 
              << std::endl ; 


 


}



void test_quad()
{
    nquad qu, qf, qi, qv ;

    qu.u = {1,1,1,1}  ;
    qf.f = {1,1,1,1} ;
    qi.i = {1,1,1,1} ;

    qu.dump("qu");
    qf.dump("qf");
    qi.dump("qi");


    int v1 = 1065353216 ; // integer behind floating point 1.

    qv.i = {v1+0,v1-1,v1+1,v1+2} ;
    qv.dump("qv");

    float ulp[4];

    ulp[0] = qv.f.x - qv.f.x ; 
    ulp[1] = qv.f.y - qv.f.x ; 
    ulp[2] = qv.f.z - qv.f.x ; 
    ulp[3] = qv.f.w - qv.f.x ; 

    printf("%.10e \n", ulp[0] );
    printf("%.10e \n", ulp[1] );
    printf("%.10e \n", ulp[2] );
    printf("%.10e \n", ulp[3] );
}


void test_make()
{
    nuvec4 u4 = make_nuvec4(0,1,2,3) ;
    u4.dump("u4");

    nivec4 i4 = make_nivec4(0,1,-2,-3) ;
    i4.dump("i4");

    nvec4 f4 = make_nvec4(0.f,1.f,-2.f,-3.f) ;
    f4.dump("f4");

    nvec3 f3 = make_nvec3(0.f,1.f,-2.f) ;
    f3.dump("f3");
}



int main()
{
    //test_quad();
    //test_make();

    test_negative_zero();

    return 0 ;
}
