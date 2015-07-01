
//  http://stackoverflow.com/questions/17064096/thrustdevice-vector-in-constant-memory

/*
   nvcc constant_memory_from_thrust_functor.cu -o /tmp/constant_memory_from_thrust_functor && /tmp/constant_memory_from_thrust_functor && rm /tmp/constant_memory_from_thrust_functor

*/


#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

const int n = 32;
__constant__ float dev_x[n]; //the array in question

struct struct_max : public thrust::unary_function<float,float> {
    float C;
    struct_max(float _C) : C(_C) {}

    // only works as a device function
    __device__ float operator()(const int& i) const { 
        // use index into constant array
        return fmax(dev_x[i],C); 
    }
};

void foo(const thrust::host_vector<float> &input_host_x, const float &x0) {
    thrust::device_vector<float> dev_sol(n);
    thrust::host_vector<float> host_sol(n);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(dev_x);
    thrust::transform(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(n),
                      dev_sol.begin(),
                      struct_max(x0));
    host_sol = dev_sol; //this line crashes

    for (int i = 0; i < n; i++)
        printf("%f\n", host_sol[i]);
}

int main() {
    thrust::host_vector<float> x(n);

    //magic happens populate x
    for (int i = 0; i < n; i++) x[i] = rand() / (float)RAND_MAX;

    cudaMemcpyToSymbol(dev_x,x.data(),n*sizeof(float));

    foo(x, 0.5);
    return(0);
}
