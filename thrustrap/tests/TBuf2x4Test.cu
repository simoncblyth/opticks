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
    ss << "$TMP/thrustrap/TBuf2x4Test/" 
       << name
       ;

    std::string s = ss.str(); 
    return strdup(s.c_str());    
}


void test_copy2x4_encapsulated()
{
    LOG(info) << "(" ;
    unsigned num_photons = 100 ; 
    unsigned hitmask = SURFACE_DETECT ; 
    unsigned modulo = 10 ; 
    unsigned num_quad = 2 ; 

    NPY<float>* way = DummyPhotonsNPY::Make(num_photons, hitmask, modulo, num_quad );
    unsigned x_num_hiy = way->getNumHit() ; 

    thrust::device_vector<float2x4> d_way(num_photons) ;  // allocate GPU buffer 
    CBufSpec cway = make_bufspec<float2x4>(d_way);        // CBufSpec holds (dec_ptr,size,num_bytes)  using thrustrap/TUtil_.cu 

    assert( cway.dev_ptr != NULL );
    assert( cway.size == num_photons );

    LOG(info) 
       << " num_photons " << num_photons
       << " sizeof(float2x4) " << sizeof(float2x4)
       << " num_photons*sizeof(float2x4) " << num_photons*sizeof(float2x4)
       << " cway.num_bytes " << cway.num_bytes
       ;

    assert( cway.num_bytes == num_photons*sizeof(float2x4) );  // <-- flakey fails, see  notes/issues/longer-thrap-tests-flakey-on-macOS.rst 

    TBuf tway("tway", cway);
    tway.upload(way);
    tway.dump2x4("tway dump2x4", 1, 0, num_photons );  // stride, begin, end 


    NPY<float>* hiy = NPY<float>::make(0,num_quad,4);

    tway.downloadSelection2x4("tway.downloadSelection2x4", hiy, hitmask );

    unsigned num_hiy = hiy->getShape(0) ;
    assert( num_hiy == x_num_hiy ); 

    const char* path = TMPPath("hiy.npy");
    hiy->save(path);
    SSys::npdump(path);
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

    test_copy2x4_encapsulated(); 

    cudaDeviceSynchronize();  
}


