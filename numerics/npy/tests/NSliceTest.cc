#include "NSlice.hpp"
#include <cstdio>
#include <vector>
#include <string>


void test_slice(const char* arg)
{
    NSlice* s = new NSlice(arg) ;
    printf("arg %s slice %s \n", arg, s->description()) ; 
}



int main()
{
    test_slice("0:10");
    test_slice("0:10:2");

    return 0 ;
}
