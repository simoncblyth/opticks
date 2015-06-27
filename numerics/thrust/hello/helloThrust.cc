
#include "hello.h"
#include "stdio.h"





int main()
{
    int v = version();
    printf("v %x\n", v);

    int hi = hello();
    printf("hi %d\n", hi);



    return 0;
}

