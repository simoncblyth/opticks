/*

::

    simon:thrust_cuda_interop blyth$ nvcc -o /tmp/device_vector_from_kernel device_vector_from_kernel.cu && /tmp/device_vector_from_kernel && rm /tmp/device_vector_from_kernel
    data = 0.000000
    data = 2.000000


*/

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

__global__ void printkernel(float *data)
{
    printf("data = %f\n", *data);
}

int main()
{
  thrust::device_vector<float> mydata(5);
  thrust::sequence(mydata.begin(), mydata.end());

  printkernel<<<1,1>>>(mydata.data().get());
  printkernel<<<1,1>>>(mydata.data().get() + 1);
  printkernel<<<1,1>>>(thrust::raw_pointer_cast(&mydata[2]));
  printkernel<<<1,1>>>(thrust::raw_pointer_cast(&mydata[2]) + 1);

  cudaDeviceSynchronize();
  return 0;
}



