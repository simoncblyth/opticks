// ./preprocessor_test.sh 
#include <cstdio>

int main(int argc, char** argv)
{
    printf("[ %s \n", argv[0] ); 
#ifdef WITH_RED
    printf("WITH_RED\n"); 
#elif WITH_GREEN   
    printf("WITH_GREEN\n"); 
#elif WITH_BLUE
    printf("WITH_BLUE\n"); 
#else
    printf("WITH ... \n"); 
#endif
    printf("]\n"); 

    return 0 ; 
}
