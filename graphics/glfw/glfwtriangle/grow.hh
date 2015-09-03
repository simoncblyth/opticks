// define away nvcc specifics : allowing struct 
#include <thrust/detail/config/host_device.h>

struct grow
{
    unsigned int count ; 
    grow(unsigned int count) : count(count) {}
    __host__ __device__ float3 operator()(unsigned int i);
};


