#include <optix.h>
#include <cstdio>

#define xstr(s) str(s)
#define str(s) #s

int main()
{
    printf("%s\n",xstr(OPTIX_VERSION)); 
    return 0 ; 
}
