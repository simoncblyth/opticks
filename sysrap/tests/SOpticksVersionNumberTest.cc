#include <cstdio>
#include "OpticksVersionNumber.hh"

int main()
{

#if OPTICKS_VERSION_NUMBER < 10
    printf("OPTICKS_VERSION_NUMBER < 10 \n"); 
#elif OPTICKS_VERSION_NUMBER == 10
    printf("OPTICKS_VERSION_NUMBER == 10 \n"); 
#elif OPTICKS_VERSION_NUMBER > 10
    printf("OPTICKS_VERSION_NUMBER > 10 \n"); 
#else
    printf("OPTICKS_VERSION_NUMBER unexpected \n"); 
#endif


#define xstr(s) str(s)
#define str(s) #s

    printf("%s\n",xstr(OPTICKS_VERSION_NUMBER)); 

    return 0 ; 
}


