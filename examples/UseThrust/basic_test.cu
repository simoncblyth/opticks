// name=basic_test ; nvcc $name.cu -std=c++11 -lstdc++ -I/usr/local/cuda/include -DWITH_THRUST -o /tmp/$name && /tmp/$name

/**

**/

#include <stdio.h>


__global__ 
void on_device(){ basic_complex::test() ; }

void on_host(){   basic_complex::test() ; }

int main()
{

#ifdef WITH_THRUST
    printf("on_device launch\n"); 
    on_device<<<1,1>>>();
    cudaDeviceSynchronize();
#endif

    printf("on_host\n"); 
    on_host(); 

    return 0 ; 
}


