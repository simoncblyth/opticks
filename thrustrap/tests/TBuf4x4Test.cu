#include "SSys.hh"

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

    NPY<float>* ph = DummyPhotonsNPY::Make(num_photons, SURFACE_DETECT );
 
    thrust::device_vector<float4x4> d_ph(num_photons) ;

    CBufSpec cph = make_bufspec<float4x4>(d_ph); 

    TBuf tph("tph", cph);

    tph.upload(ph);

    tph.dump<float4x4>("tph dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    TIsHit hitfunc ;
    unsigned numHit = thrust::count_if(d_ph.begin(), d_ph.end(), hitfunc );

    LOG(info) << "numHit :" << numHit ; 
    unsigned x_numHit = ph->getNumHit();
    assert(x_numHit == numHit );

    LOG(info) << ")" ;
}



void test_count4x4_ptr()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 

    NPY<float>* ph = DummyPhotonsNPY::Make(num_photons, SURFACE_DETECT );
 
    thrust::device_vector<float4x4> d_ph(num_photons) ;

    CBufSpec cph = make_bufspec<float4x4>(d_ph); 

    TBuf tph("tph", cph);

    tph.upload(ph);

    tph.dump<float4x4>("tph dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    thrust::device_ptr<float4x4> ptr = thrust::device_pointer_cast((float4x4*)tph.getDevicePtr()) ;

    TIsHit hitfunc ;
    unsigned numHit = thrust::count_if(ptr, ptr+num_photons, hitfunc );

    LOG(info) << "numHit :" << numHit ; 
    unsigned x_numHit = ph->getNumHit();
    assert(x_numHit == numHit );

    LOG(info) << ")" ;
}


void test_copy4x4()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 

    NPY<float>* pho = DummyPhotonsNPY::Make(num_photons, SURFACE_DETECT );
 
    thrust::device_vector<float4x4> d_pho(num_photons) ;

    CBufSpec cpho = make_bufspec<float4x4>(d_pho); 

    assert( cpho.size == num_photons );

    TBuf tpho("tpho", cpho);

    tpho.upload(pho);

    tpho.dump<float4x4>("tpho dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    TIsHit hitsel ;

    unsigned numHit = thrust::count_if(d_pho.begin(), d_pho.end(), hitsel );

    LOG(info) << "numHit :" << numHit ; 
    unsigned x_numHit = pho->getNumHit();
    assert(x_numHit == numHit );

    thrust::device_vector<float4x4> d_hit(numHit) ; 

    thrust::copy_if(d_pho.begin(), d_pho.end(), d_hit.begin(), hitsel );

    CBufSpec chit = make_bufspec<float4x4>(d_hit); 

    TBuf thit("thit", chit );  

    NPY<float>* hit = NPY<float>::make(numHit, 4,4);
    thit.download(hit);

    const char* path = "$TMP/hit.npy";
    hit->save(path);
    SSys::npdump(path);
}



void test_copy4x4_ptr()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    unsigned modulo = 10 ; 

    NPY<float>* pho = DummyPhotonsNPY::Make(num_photons, SURFACE_DETECT, modulo );

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

    TIsHit hitsel ;
    unsigned numHit = thrust::count_if(ptr, ptr+num_photons, hitsel );

    LOG(info) << "numHit :" << numHit ; 
    assert(x_numHit == numHit );

    thrust::device_vector<float4x4> d_hit(numHit) ; 

    thrust::copy_if(ptr, ptr+num_photons, d_hit.begin(), hitsel );

    CBufSpec chit = make_bufspec<float4x4>(d_hit); 


    TBuf thit("thit", chit );  

    assert(thit.getSize() == numHit );

    NPY<float>* hit = NPY<float>::make(numHit, 4,4);
    thit.download(hit);

    const char* path = "$TMP/hit.npy";
    hit->save(path);
    SSys::npdump(path);
}



void test_copy4x4_encapsulated()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    unsigned modulo = 10 ; 

    NPY<float>* pho = DummyPhotonsNPY::Make(num_photons, SURFACE_DETECT, modulo );
    unsigned x_num_hit = pho->getNumHit() ; 

    thrust::device_vector<float4x4> d_pho(num_photons) ;  // allocate GPU buffer 
    CBufSpec cpho = make_bufspec<float4x4>(d_pho);        // CBufSpec holds (dec_ptr,size,num_bytes)  using thrustrap/TUtil_.cu 

    assert( cpho.dev_ptr != NULL );
    assert( cpho.size == num_photons );
    assert( cpho.num_bytes == num_photons*sizeof(float4x4) ); 

    TBuf tpho("tpho", cpho);
    tpho.upload(pho);
    tpho.dump4x4("tpho dump4x4", 1, 0, num_photons );  // stride, begin, end 


    NPY<float>* hit = NPY<float>::make(0,4,4);
    tpho.downloadSelection4x4("tpho.downloadSelection4x4", hit );

    unsigned num_hit = hit->getShape(0) ;
    assert( num_hit == x_num_hit ); 


    const char* path = "$TMP/hit.npy";
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


