// clang fascending.c && ./a.out && rm a.out

#include "assert.h"
#include "fascending.h"

void test_fascending_ptr()
{
    float a[4] ; 

    a[0] = 3 ; 
    a[1] = 2 ; 
    a[2] = 1 ; 
    a[3] = 0 ; 

    fascending_ptr(3, a );

    assert( a[0] == 1 );
    assert( a[1] == 2 );
    assert( a[2] == 3 );


    a[0] = 10 ; 
    a[1] = 5 ; 

    fascending_ptr(2, a );

    assert( a[0] == 5 );
    assert( a[1] == 10 );
     

    a[0] = 101 ; 
    fascending_ptr(1, a );
    assert( a[0] == 101 );


    a[0] = 300 ; 
    a[1] = 200 ; 
    a[2] = 700 ; 
    a[3] = 600 ; 

    fascending_ptr(4, a );

    assert( a[0] == 200 );
    assert( a[1] == 300 );
    assert( a[2] == 600 );
    assert( a[3] == 700 );
}


int main()
{
    test_fascending_ptr() ;
    return 0 ; 
}
