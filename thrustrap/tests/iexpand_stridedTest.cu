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

//#define DEBUG 1

#include "SBuf.hh"
#include "scuda.h"

#include <cassert>
#include <sstream>
#include <string>
#include "THRAP_HEAD.hh"
#include "iexpand.h"
#include "strided_range.h"
#include <thrust/device_vector.h>
#include "THRAP_TAIL.hh"

#include <ostream>

/**
Following thrap TBufPair<T>::seedDestination
**/

void test_iexpand()
{
    int counts[] = { 3, 5, 2, 0, 1, 3, 4, 2, 4 }; 
    size_t input_size  = sizeof(counts) / sizeof(int);
    size_t output_size = thrust::reduce(counts, counts + input_size); // sum of count values


    // testing setup: in real usage the inputs will already be on device referenced via a raw device pointer 

    thrust::device_vector<int> d_counts(counts, counts + input_size);   // copy counts to device 
    thrust::device_vector<int> d_output(output_size, 0);                // prepare output buffer on device

    // test excursions via raw pointers

    thrust::device_ptr<int> psrc = d_counts.data() ;  
    int* raw_src = psrc.get(); 
    thrust::device_ptr<int> psrc2 = thrust::device_pointer_cast((int*)raw_src) ; 
    std::cout 
        << " psrc2.get() " << psrc2.get()
        << " raw_src " << raw_src
        << std::endl
        ; 
    assert( psrc2.get() == raw_src );  

    thrust::device_ptr<int> pdst = d_output.data() ;  
    int* raw_dst = pdst.get(); 
    thrust::device_ptr<int> pdst2 = thrust::device_pointer_cast((int*)raw_dst) ; 
    std::cout 
        << " pdst2.get() " << pdst2.get()
        << " raw_dst " << raw_dst
        << std::endl
        ; 
    assert( pdst2.get() == raw_dst );  

    // iexpand works with device_ptr or iterators  
    // expand 0:N-1 indices of counts according to count values
    std::cout << "[ iexpand " << std::endl ;  
    iexpand(psrc2, psrc2 + input_size, pdst2, pdst2 + output_size ); 
    std::cout << "] iexpand " << std::endl ;  


#ifdef DEBUG
    std::cout << "iExpanding indices according to counts" << std::endl;
    print(" counts ", d_counts);
    print(" output ", d_output);
#endif
}


typedef typename thrust::device_vector<int>::iterator Iterator;


void print_it( const char* msg,  strided_range<Iterator>& it )
{
    std::cout << msg << " " ;
    thrust::copy(it.begin(), it.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}


void test_strided()
{
    std::cout << "test_strided" << std::endl ; 
    int counts[] = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
 
    size_t input_size  = sizeof(counts) / sizeof(int);
    thrust::device_vector<int> d_counts(counts, counts + input_size);   // copy counts to device 
    thrust::device_ptr<int> psrc = d_counts.data() ;  

    // the device data only survives within the scope of d_counts 
    // so using device_vector is only useful for very transient device data 

    strided_range<Iterator> s1(     psrc + 0, psrc + input_size, 1 );
    strided_range<Iterator> s2even( psrc + 0, psrc + input_size, 2 );
    strided_range<Iterator> s2odd(  psrc + 1, psrc + input_size, 2 );

    print_it( "s1", s1 ); 
    print_it( "s2even", s2even ); 
    print_it( "s2odd",  s2odd  ); 

}


SBuf<int> UploadCounts()
{
    std::vector<int> counts = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    return SBuf<int>::Upload(counts) ; 
}


SBuf<int> UploadFakeGensteps_0()
{
    std::vector<int> gs ; 
    std::vector<int> counts = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    unsigned ni = counts.size(); 
    unsigned nj = 6 ; 
    unsigned nk = 4 ; 

    for(unsigned i=0 ; i < ni ; i++)
    for(unsigned j=0 ; j < nj ; j++)
    for(unsigned k=0 ; k < nk ; k++)
    {
        int value =  j == 0 && k == 3 ? counts[i] : -1 ; 
        gs.push_back(value); 
    }
    return SBuf<int>::Upload(gs) ; 
}


SBuf<quad6> UploadFakeGensteps()
{
    std::vector<quad6> gs ; 
    std::vector<int> counts = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    unsigned ni = counts.size(); 

    for(unsigned i=0 ; i < ni ; i++)
    {
        quad6 qq ; 
        qq.q0.i.x = -1 ;   qq.q0.i.y = -1 ;   qq.q0.i.z = -1 ;   qq.q0.i.w = counts[i] ; 
        qq.q1.i.x = -1 ;   qq.q1.i.y = -1 ;   qq.q1.i.z = -1 ;   qq.q1.i.w = -1 ; 
        qq.q2.i.x = -1 ;   qq.q2.i.y = -1 ;   qq.q2.i.z = -1 ;   qq.q2.i.w = -1 ; 
        qq.q3.i.x = -1 ;   qq.q3.i.y = -1 ;   qq.q3.i.z = -1 ;   qq.q3.i.w = -1 ; 
        qq.q4.i.x = -1 ;   qq.q4.i.y = -1 ;   qq.q4.i.z = -1 ;   qq.q4.i.w = -1 ; 
        qq.q5.i.x = -1 ;   qq.q5.i.y = -1 ;   qq.q5.i.z = -1 ;   qq.q5.i.w = -1 ; 
        gs.push_back(qq); 
    }

    return SBuf<quad6>::Upload(gs) ; 
}


void test_strided_scope()
{
    std::cout << "test_strided_scope" << std::endl ; 
    SBuf<int> d_counts = UploadCounts(); 
    thrust::device_ptr<int> psrc = thrust::device_pointer_cast( d_counts.ptr ) ; 
    strided_range<Iterator> s1(     psrc + 0, psrc + d_counts.num_items, 1 );
    strided_range<Iterator> s2even( psrc + 0, psrc + d_counts.num_items, 2 );
    strided_range<Iterator> s2odd(  psrc + 1, psrc + d_counts.num_items, 2 );

    print_it( "s1", s1 ); 
    print_it( "s2even", s2even ); 
    print_it( "s2odd",  s2odd  ); 
}


void test_strided_iexpand()
{
    std::cout << "test_strided_iexpand" << std::endl ; 
    SBuf<int> d_counts = UploadCounts(); 
    std::cout << " d_counts.desc " << d_counts.desc() << std::endl ; 

    thrust::device_ptr<int> psrc = thrust::device_pointer_cast( d_counts.ptr ) ; 

    size_t tot_counts = thrust::reduce(psrc, psrc + d_counts.num_items ); // sum of count values
    std::cout << "test_strided_iexpand  tot_counts " << tot_counts << std::endl ; 
    SBuf<int> d_output = SBuf<int>::Alloc(tot_counts); 
    thrust::device_ptr<int> pdst = thrust::device_pointer_cast((int*)d_output.ptr) ; 
    std::cout << " d_output.desc " << d_output.desc() << std::endl ; 


    strided_range<Iterator> s1(     psrc + 0, psrc + d_counts.num_items, 1 );
    print_it( "s1", s1 ); 

    // observe that without zeroing the output buffer in the Alloc 
    // this does not work as expected leaving the upper entries at garbage values
    //
    // THIS IS BECAUSE OF THE WAY THAT THE inclusive_scan in iexpand operates
    // IT REQUIRES THE output to start as zeroes 

    iexpand(s1.begin(), s1.end(), pdst, pdst + d_output.num_items ); 

    strided_range<Iterator> d1(     pdst + 0, pdst + d_output.num_items, 1 );
    print_it( "d1", d1 ); 
}

void test_strided_iexpand_fake_gensteps_0()
{
    std::cout << "test_strided_iexpand_fake_gensteps_0" << std::endl ; 
    SBuf<int> d_gs = UploadFakeGensteps_0() ; 
    std::cout << " d_gs.desc " << d_gs.desc() << std::endl ; 

    thrust::device_ptr<int> pgs = thrust::device_pointer_cast( d_gs.ptr ) ; 
    strided_range<Iterator> np( pgs + 3, pgs + d_gs.num_items, 6*4 );  // begin, end, stride 

    int num_photons = thrust::reduce(np.begin(), np.end() );
    std::cout << "test_strided_iexpand_fake_gensteps_0 num_photons " << num_photons << std::endl ; 

    SBuf<int> dseed = SBuf<int>::Alloc(num_photons); 
    thrust::device_ptr<int> pseed = thrust::device_pointer_cast((int*)dseed.ptr) ; 

    iexpand(np.begin(), np.end(), pseed, pseed + dseed.num_items ); 

    dseed.download_dump("dseed"); 
}


void test_strided_iexpand_fake_gensteps()
{
    std::cout << "test_strided_iexpand_fake_gensteps" << std::endl ; 

    SBuf<quad6> d_gs = UploadFakeGensteps() ; 

    std::cout << " d_gs.desc " << d_gs.desc() << std::endl ; 

    thrust::device_ptr<int> pgs = thrust::device_pointer_cast( (int*)d_gs.ptr ) ; 

    strided_range<Iterator> np( pgs + 3, pgs + d_gs.num_items*6*4 , 6*4 );  // begin, end, stride 

    int num_photons = thrust::reduce(np.begin(), np.end() );

    std::cout << "test_strided_iexpand_fake_gensteps num_photons " << num_photons << std::endl ; 

    SBuf<int> dseed = SBuf<int>::Alloc(num_photons); 

    thrust::device_ptr<int> pseed = thrust::device_pointer_cast((int*)dseed.ptr) ; 

    iexpand(np.begin(), np.end(), pseed, pseed + dseed.num_items ); 

    dseed.download_dump("dseed"); 
}


SBuf<int> create_photon_seeds( SBuf<quad6> gs )
{
    thrust::device_ptr<int> pgs = thrust::device_pointer_cast( (int*)gs.ptr ) ; 
    strided_range<Iterator> np( pgs + 3, pgs + gs.num_items*6*4, 6*4 );    // begin, end, stride 
    int num_photons = thrust::reduce(np.begin(), np.end() );
    SBuf<int> dseed = SBuf<int>::Alloc(num_photons); 
    thrust::device_ptr<int> pseed = thrust::device_pointer_cast((int*)dseed.ptr) ; 
    iexpand(np.begin(), np.end(), pseed, pseed + dseed.num_items ); 
    return dseed ; 
}



void test_create_photon_seeds()
{
    std::cout << "test_create_photon_seeds" << std::endl ; 

    SBuf<quad6> gs = UploadFakeGensteps() ; 
    //gs.download_dump("gs"); 

    SBuf<int> se = create_photon_seeds( gs ); 
    se.download_dump("se"); 
}


int main(void)
{
    //test_iexpand(); 
    //test_strided(); 
    //test_strided_scope(); 
    //test_strided_iexpand();
    //test_strided_iexpand_fake_gensteps();
    //test_strided_iexpand_fake_gensteps();
    //test_strided_iexpand_fake_gensteps_2();
    test_create_photon_seeds();
 
    return 0;
}
