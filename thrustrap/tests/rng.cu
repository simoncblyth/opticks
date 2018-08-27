#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <curand_kernel.h> 
#include <iostream>
#include <iomanip>

template<typename T>
struct PRNG
{
    typedef unsigned long long ULL ; 

    T*      dev ; 
    ULL    seed ; 
    ULL  offset ; 
    unsigned ni ; 
    unsigned nj ; 
    unsigned num_elem ; 

    __host__ 
    PRNG(T* dev_, unsigned ni_ , unsigned nj_,  ULL seed_=0ull , ULL offset_=0ull )
         : 
         dev(dev_),
         ni(ni_),
         nj(nj_),
         num_elem(ni*nj),
         seed(seed_), 
         offset(offset_)
         {}

    __device__
    void operator()(const unsigned uid ) const
    {
        curandState s;
        curand_init(seed, uid , offset, &s);

        for(unsigned j = 0; j < nj; ++j) 
        {   
            unsigned idx = uid*nj+j ;
            if(idx < num_elem )
            {   
                dev[idx] = curand_uniform(&s)  ;   
            }   
        }   
    }

    __host__
    void generate(unsigned i0, unsigned i1)
    {
        thrust::for_each(
              thrust::counting_iterator<unsigned>(i0),
              thrust::counting_iterator<unsigned>(i1),
               *this);
    }
};



int main(void)
{
    unsigned NI = 100 ;
    unsigned NJ = 16 ;
    unsigned N = NI*NJ ; 

    thrust::device_vector<float> dvec(N);
    float* udev = thrust::raw_pointer_cast(dvec.data()); 

    PRNG<float> prng(udev, NI, NJ ); 

    prng.generate(0, NI) ; 
    thrust::host_vector<float> hvec(dvec) ; 


    for( unsigned i=0 ; i < NI ; i++)
    {
        std::cout << std::setw(7) << i << " : " ; 
        for(unsigned j=0 ; j < NJ ; j++ )
             std::cout << " " << std::setw(10) << std::fixed << hvec[i*NJ+j] ; 
        std::cout << std::endl ; 
    }

    return 0;
}
