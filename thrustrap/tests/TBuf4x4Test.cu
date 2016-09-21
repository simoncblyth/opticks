#include "SSys.hh"

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>

#include "CBufSpec.hh"
#include "TBuf.hh"
#include "TUtil.hh"

#include "float4x4.h"

#include "NQuad.hpp"
#include "NPY.hpp"
#include "PLOG.hh"

// nvcc cannot stomach GLM



NPY<float>* dummy_photon_data(unsigned num_photons)
{
    unsigned PNUMQUAD = 4 ; 
    NPY<float>* photon_data = NPY<float>::make(num_photons, PNUMQUAD, 4); 
    photon_data->zero();   

    unsigned numHit(0);

    for(unsigned i=0 ; i < num_photons ; i++)
    {   
         nvec4 q0 = make_nvec4(i,i,i,i) ;
         nvec4 q1 = make_nvec4(1000+i,1000+i,1000+i,1000+i) ;
         nvec4 q2 = make_nvec4(2000+i,2000+i,2000+i,2000+i) ;

         unsigned uhit = i % 10 == 0 ? 1 : 0  ;   // one in 10 are mock "hits"  
         if(uhit > 0 ) numHit += 1 ; 

         nuvec4 u3 = make_nuvec4(3000+i,3000+i,3000+i,uhit) ;

         photon_data->setQuad( q0, i, 0 );
         photon_data->setQuad( q1, i, 1 );
         photon_data->setQuad( q2, i, 2 );
         photon_data->setQuadU( u3, i, 3 );  // flags at the end
    }   
    //photon_data->save("$TMP/ph.npy");

    photon_data->setNumHit(numHit);
    return photon_data ; 
}


void test_dump44()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    NPY<float>* ph = dummy_photon_data(num_photons);
 
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

    NPY<float>* ph = dummy_photon_data(num_photons);
 
    thrust::device_vector<float4x4> d_ph(num_photons) ;

    CBufSpec cph = make_bufspec<float4x4>(d_ph); 

    TBuf tph("tph", cph);

    tph.upload(ph);

    tph.dump<float4x4>("tph dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    LOG(info) << ")" ;
}


struct isHit : public thrust::unary_function<float4x4,bool>
{
    isHit() {}

    __host__ __device__
    bool operator()(float4x4 v)
    {   
        tquad q3 ; 
        q3.f = v.q3 ; 
        return q3.u.w > 0 ;
    }   
};


void test_count4x4()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 

    NPY<float>* ph = dummy_photon_data(num_photons);
 
    thrust::device_vector<float4x4> d_ph(num_photons) ;

    CBufSpec cph = make_bufspec<float4x4>(d_ph); 

    TBuf tph("tph", cph);

    tph.upload(ph);

    tph.dump<float4x4>("tph dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    isHit hitfunc ;
    unsigned numHit = thrust::count_if(d_ph.begin(), d_ph.end(), hitfunc );

    LOG(info) << "numHit :" << numHit ; 
    unsigned x_numHit = ph->getNumHit();
    assert(x_numHit == numHit );

    LOG(info) << ")" ;
}


void test_copy4x4()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 

    NPY<float>* pho = dummy_photon_data(num_photons);
 
    thrust::device_vector<float4x4> d_pho(num_photons) ;

    CBufSpec cpho = make_bufspec<float4x4>(d_pho); 

    TBuf tpho("tpho", cpho);

    tpho.upload(pho);

    tpho.dump<float4x4>("tpho dump<float4x4>", 1, 0, num_photons );  // stride, begin, end 

    isHit hitsel ;

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



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << argv[0] ;

    test_dump44(); 

    test_dump4x4(); 
    test_count4x4(); 
    test_copy4x4(); 

    cudaDeviceSynchronize();  
}


