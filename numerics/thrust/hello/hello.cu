#include "hello.h"


#include <thrust/version.h>
#include <iostream>

#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <thrust/sort.h>


int version()
{
   int major = THRUST_MAJOR_VERSION;
   int minor = THRUST_MINOR_VERSION;
   std::cout << "Thrust v" << major << "." << minor << std::endl; 
   return major << 8 | minor ;
}

int hello()
{
    int n = 1 << 24 ; 
    thrust::host_vector<int> h_vec(n); 
    thrust::generate(h_vec.begin(), h_vec.end(), rand);
    thrust::device_vector<int> d_vec = h_vec; // sort data on the device
    thrust::sort(d_vec.begin(), d_vec.end()); // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin()); 

    return n ;
}
