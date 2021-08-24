#include <cstdio>
#include "OpticksVersionNumber.hh"

#define xstr(s) str(s)
#define str(s) #s

int main()
{
    printf("%s\n",xstr(OPTICKS_VERSION_NUMBER)); 
    return 0 ; 
}


