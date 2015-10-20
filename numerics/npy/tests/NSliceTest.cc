#include "NSlice.hpp"
#include <cstdio>
#include <vector>
#include <string>


int main(int argc, char** argv)
{
    for(int i=1 ; i < argc ; i++)
    {
        char* arg = argv[i] ;
        NSlice* s = new NSlice(arg) ;
        printf("arg %s slice %s \n", arg, s->description()) ; 
    }
    return 0 ;
}
