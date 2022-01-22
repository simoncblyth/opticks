// name=f4_stack_test ; gcc $name.cc -I.. -I$OPTICKS_PREFIX/include/SysRap -I/usr/local/cuda/include  -o /tmp/$name  && /tmp/$name

#include "stdio.h"
#include "assert.h"

#include "scuda.h"
#include "f4_stack.h"

void test_1( F4_Stack& s )
{
    printf("//test_1\n"); 

    float a0, a1 ;
 
    for(int i=0 ; i < 10 ; i++)
    {
       a0 = float(i); 
       s.push(a0) ; 
       s.pop(a1); 
       printf("  a0 %10.4f a1 %10.4f \n", a0, a1 ); 
       assert( a0 == a1 ); 
    }

    assert( s.curr == -1 ); 
}

void test_2( F4_Stack& s, int mode )
{
    printf("//test_2\n"); 
    float a0, b0 ; 
    float a1, b1 ;
 
    for(int i=0 ; i < 10 ; i++)
    {
       a0 = float(i); 
       b0 = float(10*i); 

       if( mode == 1 )
       {
           s.push(a0) ; 
           s.push(b0) ; 

           s.pop(b1);
           s.pop(a1);
       }
       else if( mode == 2)
       {
           s.push2(a0, b0); 
           s.pop2( a1, b1); 
       }
 
       printf("  mode %d a0 %10.4f b0 %10.4f b1 %10.4f a1 %10.4f  \n", mode, a0, b0, b1, a1 ); 
       assert( b1 == b0 );
       assert( a1 == a0 );
    }
    assert( s.curr == -1 ); 
}

void test_3( F4_Stack& s, int mode)
{
    printf("//test_3\n"); 
    float a0, b0, c0 ; 
    float a1, b1, c1 ; 
 
    for(int i=0 ; i < 10 ; i++)
    {
       a0 = float(i); 
       b0 = float(10*i); 
       c0 = float(100*i); 

       if( mode == 1 )
       {
           s.push(a0) ; 
           s.push(b0) ; 
           s.push(c0) ; 

           s.pop(c1);
           s.pop(b1);
           s.pop(a1);
       }
       else if( mode == 2 )
       {
           s.push2(a0, b0); 
           s.push(c0); 

           s.pop(c1); 
           s.pop2(a1,b1); 
       }

 
       printf(" mode %d  a0 %10.4f b0 %10.4f c0 %10.4f c1 %10.4f b1 %10.4f a1 %10.4f  \n", mode, a0, b0, c0, c1, b1, a1 ); 
       assert( c1 == c0 );
       assert( b1 == b0 );
       assert( a1 == a0 );
    }
    assert( s.curr == -1 ); 
}

void test_4( F4_Stack& s, int mode )
{
    printf("//test_4\n"); 
    float a0, b0, c0, d0 ;
    float a1, b1, c1, d1 ;
 
    for(int i=0 ; i < 10 ; i++)
    {
       a0 = float(i); 
       b0 = float(10*i); 
       c0 = float(100*i); 
       d0 = float(1000*i); 

       if( mode == 1 )
       {
           s.push(a0) ; 
           s.push(b0) ; 
           s.push(c0) ; 
           s.push(d0) ; 

           s.pop(d1);
           s.pop(c1);
           s.pop(b1);
           s.pop(a1);
       }
       else if( mode == 2 )
       {
           s.push2( a0, b0 ); 
           s.push2( c0, d0 ); 
           s.pop2(  c1, d1 ); 
           s.pop2(  a1, b1 ); 
       }
 
       printf(" mode %d a0 %10.4f b0 %10.4f c0 %10.4f d0 %10.4f d1 %10.4f c1 %10.4f b1 %10.4f a1 %10.4f  \n", mode, a0, b0, c0, d0, d1, c1, b1, a1 ); 

       assert( d1 == d0 );
       assert( c1 == c0 );
       assert( b1 == b0 );
       assert( a1 == a0 );
    }
    assert( s.curr == -1 ); 
}


int main(int argc, char** argv)
{
    printf("%s\n", argv[0] ); 

    F4_Stack s ; 
    s.curr = -1 ; 

    test_1(s); 

    test_2(s, 1);
    test_2(s, 2);
 
    test_3(s, 1);
    test_3(s, 2);
 
    test_4(s, 1);
    test_4(s, 2);

    return 0 ; 
}
