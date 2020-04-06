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

#include "SSys.hh"

#include <string>
#include <sstream>

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>

#include "CBufSpec.hh"
#include "TBuf.hh"
#include "TUtil.hh"

#include "TIsHit.hh"
#include "float4x4.h"

#include "OpticksPhoton.h"
#include "DummyPhotonsNPY.hpp"
#include "NPY.hpp"
#include "OPTICKS_LOG.hh"

// nvcc cannot stomach GLM


const char* TMPPath( const char* name)
{
    std::stringstream ss ;
    ss << "$TMP/thrustrap/TBuf4x4Test/" 
       << name
       ;

    std::string s = ss.str(); 
    return strdup(s.c_str());    
}



void test_dump44()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    NPY<float>* ph = DummyPhotonsNPY::Make(num_photons, SURFACE_DETECT );
 
    thrust::device_vector<float4> d_ph(num_photons*4) ;

    CBufSpec cph = make_bufspec<float4>(d_ph); 

    TBuf tph("tph", cph);

    tph.upload(ph);

    tph.dump<float4>("tph dump<float4>", 1, 0, num_photons*4 );  // stride, begin, end 
    tph.dump<float4x4>("tph dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    LOG(info) << ")" ;
}


void test_dump4x4()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 

    NPY<float>* ph = DummyPhotonsNPY::Make(num_photons, SURFACE_DETECT );
 
    thrust::device_vector<float4x4> d_ph(num_photons) ;

    CBufSpec cph = make_bufspec<float4x4>(d_ph); 

    TBuf tph("tph", cph);

    tph.upload(ph);

    tph.dump<float4x4>("tph dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    LOG(info) << ")" ;
}



void test_count4x4()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    unsigned hitmask = SURFACE_DETECT ; 

    NPY<float>* ph = DummyPhotonsNPY::Make(num_photons, hitmask );
 
    thrust::device_vector<float4x4> d_ph(num_photons) ;

    CBufSpec cph = make_bufspec<float4x4>(d_ph); 

    TBuf tph("tph", cph);

    tph.upload(ph);

    tph.dump<float4x4>("tph dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 


    TIsHit is_hit(hitmask) ;
    unsigned numHit = thrust::count_if(d_ph.begin(), d_ph.end(), is_hit );

    LOG(info) << "numHit :" << numHit ; 
    unsigned x_numHit = ph->getNumHit();
    assert(x_numHit == numHit );

    LOG(info) << ")" ;
}



void test_count4x4_ptr()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    unsigned hitmask = SURFACE_DETECT ; 

    NPY<float>* ph = DummyPhotonsNPY::Make(num_photons, hitmask );
 
    thrust::device_vector<float4x4> d_ph(num_photons) ;

    CBufSpec cph = make_bufspec<float4x4>(d_ph); 

    TBuf tph("tph", cph);

    tph.upload(ph);

    tph.dump<float4x4>("tph dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    thrust::device_ptr<float4x4> ptr = thrust::device_pointer_cast((float4x4*)tph.getDevicePtr()) ;

    TIsHit is_hit(hitmask) ;
    unsigned numHit = thrust::count_if(ptr, ptr+num_photons, is_hit );

    LOG(info) << "numHit :" << numHit ; 
    unsigned x_numHit = ph->getNumHit();
    assert(x_numHit == numHit );

    LOG(info) << ")" ;
}


void test_copy4x4()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    unsigned hitmask = SURFACE_DETECT ; 
    NPY<float>* pho = DummyPhotonsNPY::Make(num_photons, hitmask );
 
    thrust::device_vector<float4x4> d_pho(num_photons) ;

    CBufSpec cpho = make_bufspec<float4x4>(d_pho); 

    assert( cpho.size == num_photons );

    TBuf tpho("tpho", cpho);

    tpho.upload(pho);

    tpho.dump<float4x4>("tpho dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    TIsHit is_hit(hitmask) ;

    unsigned numHit = thrust::count_if(d_pho.begin(), d_pho.end(), is_hit );

    LOG(info) << "numHit :" << numHit ; 
    unsigned x_numHit = pho->getNumHit();
    assert(x_numHit == numHit );

    thrust::device_vector<float4x4> d_hit(numHit) ; 

    thrust::copy_if(d_pho.begin(), d_pho.end(), d_hit.begin(), is_hit );

    CBufSpec chit = make_bufspec<float4x4>(d_hit); 

    TBuf thit("thit", chit );  

    NPY<float>* hit = NPY<float>::make(numHit, 4,4);
    thit.download(hit);

    const char* path = TMPPath("hit.npy");
    hit->save(path);
    SSys::npdump(path);
}



void test_copy4x4_ptr()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    unsigned hitmask = SURFACE_DETECT ; 
    unsigned modulo = 10 ; 

    NPY<float>* pho = DummyPhotonsNPY::Make(num_photons, hitmask, modulo );

    unsigned x_numHit = pho->getNumHit();
 
    thrust::device_vector<float4x4> d_pho(num_photons) ;

    CBufSpec cpho = make_bufspec<float4x4>(d_pho); 

    assert( cpho.size == num_photons );

    // check can operate from TBuf alone, without help from device_vector

    TBuf tpho("tpho", cpho);

    tpho.upload(pho);

    tpho.dump<float4x4>("tpho dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 


    thrust::device_ptr<float4x4> ptr = thrust::device_pointer_cast((float4x4*)tpho.getDevicePtr()) ;

    assert(num_photons == tpho.getSize());

    TIsHit is_hit(hitmask) ;
    unsigned numHit = thrust::count_if(ptr, ptr+num_photons, is_hit );

    LOG(info) << "numHit :" << numHit ; 
    assert(x_numHit == numHit );

    thrust::device_vector<float4x4> d_hit(numHit) ; 

    thrust::copy_if(ptr, ptr+num_photons, d_hit.begin(), is_hit );

    CBufSpec chit = make_bufspec<float4x4>(d_hit); 


    TBuf thit("thit", chit );  

    assert(thit.getSize() == numHit );

    NPY<float>* hit = NPY<float>::make(numHit, 4,4);
    thit.download(hit);

    const char* path = TMPPath("hit.npy");
    hit->save(path);
    SSys::npdump(path);
}



void test_copy4x4_encapsulated()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    unsigned hitmask = SURFACE_DETECT ; 
    unsigned modulo = 10 ; 

    NPY<float>* pho = DummyPhotonsNPY::Make(num_photons, hitmask, modulo );
    unsigned x_num_hit = pho->getNumHit() ; 

    thrust::device_vector<float4x4> d_pho(num_photons) ;  // allocate GPU buffer 
    CBufSpec cpho = make_bufspec<float4x4>(d_pho);        // CBufSpec holds (dec_ptr,size,num_bytes)  using thrustrap/TUtil_.cu 

    assert( cpho.dev_ptr != NULL );
    assert( cpho.size == num_photons );

    LOG(info) 
       << " num_photons " << num_photons
       << " sizeof(float4x4) " << sizeof(float4x4)
       << " num_photons*sizeof(float4x4) " << num_photons*sizeof(float4x4)
       << " cpho.num_bytes " << cpho.num_bytes
       ;

    assert( cpho.num_bytes == num_photons*sizeof(float4x4) );  // <-- flakey fails, see  notes/issues/longer-thrap-tests-flakey-on-macOS.rst 

    TBuf tpho("tpho", cpho);
    tpho.upload(pho);
    tpho.dump4x4("tpho dump4x4", 1, 0, num_photons );  // stride, begin, end 


    NPY<float>* hit = NPY<float>::make(0,4,4);

    tpho.downloadSelection4x4("tpho.downloadSelection4x4", hit, hitmask );

    unsigned num_hit = hit->getShape(0) ;
    assert( num_hit == x_num_hit ); 


    const char* path = TMPPath("hit.npy");
    hit->save(path);
    SSys::npdump(path);
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

/*
    test_dump44(); 

    test_dump4x4(); 
    test_count4x4(); 
    test_copy4x4(); 


    test_count4x4_ptr(); 
*/

    test_copy4x4_ptr(); 
    test_copy4x4_encapsulated(); 

    cudaDeviceSynchronize();  
}


