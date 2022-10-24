#include "basic_complex.h"

void on_host(){   basic_complex::test() ; }

int main()
{
#ifdef WITH_THRUST
    printf("on_host WITH_THRUST\n"); 
#else
    printf("on_host NOT WITH_THRUST\n"); 
#endif
    on_host(); 

    return 0 ; 
}

