// name=cuDeviceGetCount ; tmp=/tmp/$USER/opticks ; mkdir -p $tmp ; gcc $name.cc -lstdc++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -o $tmp/$name && $tmp/$name
#include <cuda.h>
#include <iostream>

int main(int argc, char** argv)
{
    int deviceCount(-1) ;
    cuDeviceGetCount(&deviceCount);
    std::cout << "deviceCount " << deviceCount << std::endl ; 
    return 0 ; 
}
