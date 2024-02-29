#include <optix.h>
#include <cstdio>

#define xstr(s) str(s)
#define str(s) #s

int main()
{
    const char* vers = xstr(OPTIX_VERSION) ; 

#if OPTIX_VERSION < 70000
    printf("Got unexpected version < 70000 :  %s\n",vers); 
#elif OPTIX_VERSION == 70000
    printf("Got 70000 : vers %s \n", vers ); 
#elif OPTIX_VERSION == 70200
    printf("Got 70200 : vers %s \n", vers ); 
#elif OPTIX_VERSION == 70300
    printf("Got 70300 : vers %s \n", vers ); 
#elif OPTIX_VERSION == 70400
    printf("Got 70400 : vers %s \n", vers ); 
#elif OPTIX_VERSION == 70500
    printf("Got 70500 : vers %s \n", vers ); 
#elif OPTIX_VERSION == 70600
    printf("Got 70600 : vers %s \n", vers ); 
#elif OPTIX_VERSION == 70700
    printf("Got 70700 : vers %s \n", vers ); 
#elif OPTIX_VERSION == 80000
    printf("Got 80000 : vers %s \n", vers ); 
#else
    printf("Got unexpected version %s\n", vers ); 
#endif 

    return 0 ; 
}


