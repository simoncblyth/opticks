/**
name=crovella_t66 ; nvcc $name.cu -o /tmp/$name && /tmp/$name

https://stackoverflow.com/questions/64464892/how-to-use-thrustcopy-if-using-pointers

**/

#include <thrust/copy.h>
#include <iostream>
#include <iomanip>
#include <thrust/device_ptr.h>

struct is_not_zero
{
    __host__ __device__
    bool operator()( double x)
    {
        return (x != 0);
    }
};


int main()
{
    const int ds = 20 ;

    float src[ds];
    for (int i = 0; i < ds; i++) src[i] = i % 3 == 0 ? 0.f : 100.5f + float(i)  ;

    float* d_src ;
    cudaMalloc(&d_src, ds*sizeof(float));
    cudaMemcpy(d_src, src, ds*sizeof(float), cudaMemcpyHostToDevice);
    thrust::device_ptr<float> t_src(d_src);

    is_not_zero predicate ; 
    size_t num_sel = thrust::count_if(t_src, t_src+ds, predicate );
    std::cout << " num_sel " << num_sel << std::endl ; 

    float* dst = new float[num_sel] ; 
    for (int i = 0; i < num_sel ; i++) dst[i] = 0.f ;

    float* d_dst ;  
    cudaMalloc(&d_dst,        num_sel*sizeof(float));
    cudaMemset(d_dst, 0,      num_sel*sizeof(float));
    thrust::device_ptr<float> t_dst(d_dst);   // thrust::device_pointer_cast(d_dst);

    thrust::copy_if(t_src, t_src+ds, t_dst, predicate );

    cudaMemcpy(dst, d_dst, num_sel*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<ds;i++) 
        std::cout 
            << " src " << std::setw(10) << ( i < ds ? src[i] : -1.f )  
            << " dst " << std::setw(10) << ( i < num_sel ? dst[i] : -1.f ) 
            << std::endl 
            ;

}
