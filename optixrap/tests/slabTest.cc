

#include <cstdio>
#include <cmath>


void test_mm(float a, float b)
{
    printf("fmaxf(%f,%f)    = %f\n", a,b,fmaxf(a,b));
    printf("fminf(%f,%f)    = %f\n", a,b,fminf(a,b));
}

 
int main(void)
{

    float a = 1.f/0.f ; 
    float b = 1.f ; 

    test_mm( 1.f/0.f, 1.f );
    test_mm( 1.f    , 1.f/0.f );

    test_mm(  1.f/0.f, -1.f/0.f );
    test_mm( -1.f/0.f,  1.f/0.f );



    return 0 ; 
}
