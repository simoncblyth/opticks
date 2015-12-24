#include <thrust/for_each.h>
#include <thrust/device_vector.h>

struct printf_functor_i
{
  __host__ __device__
  void operator()(int x)
  {
    printf("%d\n", x); 
  }
};


struct printf_functor_f4
{
  __host__ __device__
  void operator()(float4 v)
  {
    printf("%10.4f %10.4f %10.4f %10.4f \n", v.x, v.y, v.z, v.w);
  }
};


int main()
{
    thrust::device_vector<int> ivec(3);
    ivec[0] = 0;  
    ivec[1] = 1;  
    ivec[2] = 2;
    thrust::for_each(ivec.begin(), ivec.end(), printf_functor_i());


    thrust::device_vector<float4> fvec(3);
    fvec[0] = make_float4( 1.f, 2.f, 3.f, 4.f );  
    fvec[1] = make_float4( 1.f, 2.f, 3.f, 4.f );  
    fvec[2] = make_float4( 1.f, 2.f, 3.f, 4.f );  

    thrust::for_each(fvec.begin(), fvec.end(), printf_functor_f4());


    cudaDeviceSynchronize();  

    // Without the sync the process will typically terminate before 
    // any output stream gets pumped out to the terminal when 
    // iterating over device_ptr. 
    // Curiously that doesnt seem to happen with device_vector ? 
    // Maybe their dtors are delayed by the dumping
}
