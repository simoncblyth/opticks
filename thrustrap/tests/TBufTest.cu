/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <thrust/for_each.h>
#include <thrust/device_vector.h>

#include "CBufSpec.hh"
#include "TBuf.hh"
#include "TUtil.hh"

#include "NPY.hpp"
#include "PLOG.hh"


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
    LOG(info) << "(" ;
    thrust::device_vector<int> ivec(3);
    ivec[0] = 0;  
    ivec[1] = 1;  
    ivec[2] = 2;
    thrust::for_each(ivec.begin(), ivec.end(), printf_functor_i());
    LOG(info) << ")" ;
}


void test_cbufspec()
{
    LOG(info) << "(" ;
    thrust::device_vector<int> ivec(3);
    ivec[0] = 0;  
    ivec[1] = 1;  
    ivec[2] = 2;

    CBufSpec ibs = make_bufspec<int>(ivec);
    ibs.Summary("ibs"); 
    LOG(info) << ")" ;
}

void test_tbuf(unsigned n, unsigned stride)
{
    LOG(info) << "(" ;
    thrust::device_vector<int> ivec(n);

    for(unsigned i=0 ; i < n ; i++) ivec[i] = i ; 

    CBufSpec ibs = make_bufspec<int>(ivec);
    ibs.Summary("ibs"); 

    TBuf tibs("tibs", ibs );
    tibs.dump<int>("tibs dump", stride, 0, n ); // stride, begin, end

    LOG(info) << ")" ;
}

void test_ull(unsigned int n, unsigned stride)
{
    LOG(info) << "(" ;
    thrust::device_vector<unsigned long long> uvec(n);

    for(unsigned i=0 ; i < n ; i++)
    {
        unsigned j = i % 3 ; 
        if(      j == 0) uvec[i] = 0xffeedd;  
        else if( j == 1) uvec[i] = 0xffaabb;  
        else if( j == 2) uvec[i] = 0xffbbcc;
        else             uvec[i] = 0xffffff;
    }
 
    //thrust::for_each(ivec.begin(), ivec.end(), printf_functor_i());

    CBufSpec ubs = make_bufspec<unsigned long long>(uvec);
    ubs.Summary("ubs"); 

    TBuf tubs("tubs", ubs );
    tubs.dump<unsigned long long>("tubs dump", stride, 0, n ); 
    LOG(info) << ")" ;
}


void test_f4()
{
    LOG(info) << "(" ;
    thrust::device_vector<float4> fvec(3);
    fvec[0] = make_float4( 1.f, 2.f, 3.f, 4.f );  
    fvec[1] = make_float4( 1.f, 2.f, 3.f, 4.f );  
    fvec[2] = make_float4( 1.f, 2.f, 3.f, 4.f );  

    thrust::for_each(fvec.begin(), fvec.end(), printf_functor_f4());

    CBufSpec fbs = make_bufspec<float4>(fvec);
    fbs.Summary("fbs"); 
    LOG(info) << ")" ;
}



void test_dump0()
{

    const char* pfx = NULL ;  
    LOG(info) << "(" ;
    NPY<unsigned long long>* ph = NPY<unsigned long long>::load(pfx, "ph%s", "torch",  "-5", "rainbow" );
    // check 
    if (!ph) {
        printf("can't load data\n");
        return  ;
    }


    thrust::device_vector<unsigned long long> d_ph(ph->begin(), ph->end());

    CBufSpec cph = make_bufspec<unsigned long long>(d_ph); 

    TBuf tph("tph", cph);

    tph.dump<unsigned long long>("tph dump", 2, 0, 10 ); 

    LOG(info) << ")" ;
}




void test_dump()
{
    LOG(info) << "(" ;
    NPY<unsigned long long>* ph = NPY<unsigned long long>::make(100);
    ph->zero(); 

    thrust::device_vector<unsigned long long> d_ph(ph->begin(), ph->end());

    CBufSpec cph = make_bufspec<unsigned long long>(d_ph); 

    TBuf tph("tph", cph);

    tph.dump<unsigned long long>("tph dump", 2, 0, 10 ); 

    LOG(info) << ")" ;
}


void test_download()
{
    LOG(info) << "(" ;
    NPY<unsigned long long>* ph = NPY<unsigned long long>::make(100);
    ph->zero(); 

    unsigned long long one = 1ull ; 
    ph->fill(one); 

    thrust::device_vector<unsigned long long> d_ph(ph->begin(), ph->end());

    CBufSpec cph = make_bufspec<unsigned long long>(d_ph); 

    TBuf tph("tph", cph);

    tph.dump<unsigned long long>("tph dump", 2, 0, 10 ); 

    bool verbose = true ; 

    tph.download( ph, verbose ); 


    LOG(info) << ")" ;
}











int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << argv[0] ;

/*
    test_foreach();
    test_cbufspec();

    unsigned stride = 1 ; 

    test_tbuf(3,stride);
    test_tbuf(4,stride);
    test_ull(3,stride);
    test_ull(6,stride);

    test_f4();
    test_dump(); 
*/
    test_download(); 


    cudaDeviceSynchronize();  
}

// Without the sync the process will typically terminate before 
// any output stream gets pumped out to the terminal when 
// iterating over device_ptr. 
// Curiously that doesnt seem to happen with device_vector ? 
// Maybe their dtors are delayed by the dumping


/*

simonblyth@optix thrustrap]$ TBufTest
2016-07-08 17:56:01.307 INFO  [32347] [main@140] TBufTest
2016-07-08 17:56:01.307 INFO  [32347] [test_foreach@36] (
2016-07-08 17:56:01.592 INFO  [32347] [test_foreach@42] )
0
1
2
2016-07-08 17:56:01.593 INFO  [32347] [test_cbufspec@48] (
ibs : dev_ptr 0xb07200000 size 3 num_bytes 12 
2016-07-08 17:56:01.593 INFO  [32347] [test_cbufspec@56] )
2016-07-08 17:56:01.593 INFO  [32347] [test_tbuf@61] (
ibs : dev_ptr 0xb07200000 size 3 num_bytes 12 
tibs dump tibs 
terminate called after throwing an instance of 'thrust::system::system_error'
  what():  function_attributes(): after cudaFuncGetAttributes: invalid device function
Aborted (core dumped)


*/


