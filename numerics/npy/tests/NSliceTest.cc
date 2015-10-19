#include "NSlice.hpp"
#include <cstdio>

int main()
{
    NSlice* s = new NSlice("50:100:2") ;
    printf("%s \n", s->description()) ; 


    return 0 ;
}
