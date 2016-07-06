#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include "CBufSpec.hh"
#include "TBuf.hh"
#include "TUtil.hh"

#include "NPY.hpp"


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



void test_foreach()
{
    thrust::device_vector<int> ivec(3);
    ivec[0] = 0;  
    ivec[1] = 1;  
    ivec[2] = 2;
    thrust::for_each(ivec.begin(), ivec.end(), printf_functor_i());
}


void test_cbufspec()
{
    thrust::device_vector<int> ivec(3);
    ivec[0] = 0;  
    ivec[1] = 1;  
    ivec[2] = 2;

    CBufSpec ibs = make_bufspec<int>(ivec);
    ibs.Summary("ibs"); 
}

void test_tbuf()
{
    thrust::device_vector<int> ivec(3);
    ivec[0] = 0;  
    ivec[1] = 1;  
    ivec[2] = 2;

    CBufSpec ibs = make_bufspec<int>(ivec);
    ibs.Summary("ibs"); 

    TBuf tibs("tibs", ibs );
    tibs.dump<int>("tibs dump", 1, 0, 3 ); 
}

void test_ull()
{
    thrust::device_vector<unsigned long long> uvec(3);
    uvec[0] = 0xffeedd;  
    uvec[1] = 0xffaabb;  
    uvec[2] = 0xffbbcc;
    //thrust::for_each(ivec.begin(), ivec.end(), printf_functor_i());

    CBufSpec ubs = make_bufspec<unsigned long long>(uvec);
    ubs.Summary("ubs"); 

    TBuf tubs("tubs", ubs );
    tubs.dump<unsigned long long>("tubs dump", 1, 0, 3 ); 
}

void test_f4()
{
    thrust::device_vector<float4> fvec(3);
    fvec[0] = make_float4( 1.f, 2.f, 3.f, 4.f );  
    fvec[1] = make_float4( 1.f, 2.f, 3.f, 4.f );  
    fvec[2] = make_float4( 1.f, 2.f, 3.f, 4.f );  

    thrust::for_each(fvec.begin(), fvec.end(), printf_functor_f4());

    CBufSpec fbs = make_bufspec<float4>(fvec);
    fbs.Summary("fbs"); 
}



int main()
{
    NPY<unsigned long long>* ph = NPY<unsigned long long>::load("ph%s", "torch",  "-5", "rainbow" );
    // check 
    if (!ph) {
        printf("can't load data\n");
        return -1;
    }

    thrust::device_vector<unsigned long long> d_ph(ph->begin(), ph->end());

    CBufSpec cph = make_bufspec<unsigned long long>(d_ph); 

    TBuf tph("tph", cph);

    tph.dump<unsigned long long>("tph dump", 2, 0, 10 ); 


    cudaDeviceSynchronize();  
}

// Without the sync the process will typically terminate before 
// any output stream gets pumped out to the terminal when 
// iterating over device_ptr. 
// Curiously that doesnt seem to happen with device_vector ? 
// Maybe their dtors are delayed by the dumping

