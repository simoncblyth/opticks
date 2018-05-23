#include "NSlice.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <cassert>


void test_slice(const char* arg)
{
    NSlice* s = new NSlice(arg) ;
    printf("arg %s slice %s \n", arg, s->description()) ; 
}

void test_margin()
{
    NSlice s(0,10,1);

    assert(  s.isHead(0,2) );
    assert(  s.isHead(1,2) );
    assert( !s.isHead(2,2) );
    assert( !s.isHead(3,2) );
    assert( !s.isHead(4,2) );
    assert( !s.isHead(5,2) );
    assert( !s.isHead(6,2) );
    assert( !s.isHead(7,2) );
    assert( !s.isHead(8,2) );
    assert( !s.isHead(9,2) );
    assert( !s.isHead(10,2) );

    assert( !s.isTail(0,2) );
    assert( !s.isTail(1,2) );
    assert( !s.isTail(2,2) );
    assert( !s.isTail(3,2) );
    assert( !s.isTail(4,2) );
    assert( !s.isTail(5,2) );
    assert( !s.isTail(6,2) );
    assert( !s.isTail(7,2) );
    assert(  s.isTail(8,2) );
    assert(  s.isTail(9,2) );
    assert( !s.isTail(10,2) );

    assert(  s.isMargin(0,2) );
    assert(  s.isMargin(1,2) );
    assert( !s.isMargin(2,2) );
    assert( !s.isMargin(3,2) );
    assert( !s.isMargin(4,2) );
    assert( !s.isMargin(5,2) );
    assert( !s.isMargin(6,2) );
    assert( !s.isMargin(7,2) );
    assert(  s.isMargin(8,2) );
    assert(  s.isMargin(9,2) );
    assert( !s.isMargin(10,2) );
}



int main()
{
    test_slice("0:10");
    test_slice("0:10:2");

    test_margin();

    return 0 ;
}
