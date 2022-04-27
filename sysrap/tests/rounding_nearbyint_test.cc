// name=rounding_nearbyint_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

/**

https://en.cppreference.com/w/cpp/numeric/math/nearbyint

https://en.cppreference.com/w/cpp/numeric/math/rint

**/

#include <cassert>
#include <cstdio>
#include <cmath>
#include <iostream>


void test_char()
{
    // char can hold values from -128 to 127  (char cannot hold 128, that would wrap to -128)
    for(int i=-128 ; i < 128 ; i++) printf("// i %d  (char) %d \n", i, int((char)i) );   
}

void test_lrint()
{
    // compressing a value known to be in range -1.f -> 1.f (eg polarization component)
    float df = 1.f/127.f ; 
    unsigned count = 0 ;  
    for(float f=-1.f ; f <= 1.f ; f+=df )
    {
        float ff = (f+1.f)*127.f ; 
        long l = lrint(ff) ;  
        unsigned char uc = l ; 
        printf("// count %3d  f %10.4f  ff %10.4f  l %3ld uc %3d  \n", count, f,ff, l, uc  ); 
        assert( count == unsigned(l) ); 
        count += 1 ; 
    } 
    printf("// df %10.4f \n", df ); 
}

int main()
{
    test_lrint(); 
    return 0 ; 
}
