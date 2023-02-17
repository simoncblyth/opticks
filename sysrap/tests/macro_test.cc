// name=macro_test ; gcc $name.cc -std=c++11  -lstdc++ -DVERSION=70000 -o /tmp/$name && /tmp/$name

#include <cstdio>

int main()
{
#if VERSION == 70000 
    printf("70000\n"); 
#elif VERSION == 70500 || VERSION == 70600
    printf("70500 || 70600 \n"); 
#else
    printf("OTHER\n"); 
#endif

    return 0 ; 
}
