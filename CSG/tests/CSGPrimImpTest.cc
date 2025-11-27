#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>

#ifdef WITH_CUDA
#include "cuda.h"
#include "CU.h"
#endif

#include "scuda.h"
#include "CSGPrim.h"


CSGPrim make_prim( float extent, unsigned idx )
{
     CSGPrim pr = {} ;
     pr.setAABB( extent );
     pr.setSbtIndexOffset(idx);
     return pr ;
}



void test_AABB()
{
    CSGPrim p0 = {} ;
    p0.setAABB( 42.42f );
    std::cout << "p0 " << p0.desc() << std::endl ;

    CSGPrim p1 = {} ;
    p1.setAABB( p0.AABB() );
    std::cout << "p1 " << p1.desc() << std::endl ;
}



void test_offsets()
{
     std::cout << "test_offsets " << std::endl ;
     std::cout
        <<  "offsetof(struct CSGPrim, q0) " <<  offsetof(struct CSGPrim,  q0) << std::endl
        <<  "offsetof(struct CSGPrim, q0)/sizeof(float) " <<  offsetof(struct CSGPrim, q0)/sizeof(float) << std::endl
        ;
     std::cout
        <<  "offsetof(struct CSGPrim, q1) " <<  offsetof(struct CSGPrim,  q1) << std::endl
        <<  "offsetof(struct CSGPrim, q1)/sizeof(float) " <<  offsetof(struct CSGPrim, q1)/sizeof(float) << std::endl
        ;

     std::cout
        <<  "offsetof(struct CSGPrim, q2) " <<  offsetof(struct CSGPrim,  q2) << std::endl
        <<  "offsetof(struct CSGPrim, q2)/sizeof(float) " <<  offsetof(struct CSGPrim, q2)/sizeof(float) << std::endl
        ;
     std::cout
        <<  "offsetof(struct CSGPrim, q3) " <<  offsetof(struct CSGPrim,  q3) << std::endl
        <<  "offsetof(struct CSGPrim, q3)/sizeof(float) " <<  offsetof(struct CSGPrim, q3)/sizeof(float) << std::endl
        ;
}


void test_spec( const std::vector<CSGPrim>& prim )
{
     std::cout << "test_spec " << std::endl ;
     SCSGPrimSpec psa = CSGPrim::MakeSpec(prim.data(), 0, prim.size() );
     psa.dump();

     std::vector<float> out ;
     psa.gather(out);
     SCSGPrimSpec::Dump(out);
}

void test_partial( const std::vector<CSGPrim>& prim )
{
     std::cout << "test_partial " << std::endl ;
     unsigned h = prim.size()/2 ;

     SCSGPrimSpec ps0 = CSGPrim::MakeSpec(prim.data(), 0, h );
     ps0.dump();

     SCSGPrimSpec ps1 = CSGPrim::MakeSpec(prim.data(), h, h );
     ps1.dump();
}

#ifdef WITH_CUDA
CSGPrim* test_upload( const CSGPrim* prim, unsigned num )
{
     std::cout << "test_upload" << std::endl ;
     CSGPrim* d_prim = CU::UploadArray<CSGPrim>(prim, num ) ;
     assert( d_prim );
     return d_prim ;
}
void test_download( const CSGPrim* d_prim, unsigned num )
{
     std::cout << "test_download" << std::endl ;
     CSGPrim* prim2 = CU::DownloadArray<CSGPrim>( d_prim,  num ) ;
     for(unsigned i=0 ; i < num ; i++)
     {
         CSGPrim* p = prim2 + i  ;
         std::cout << i << std::endl << p->desc() << std::endl ;
     }
}

/**
Highly inefficienct noddy appoach as not worth getting into
thrust (or cudaMemcpy2D) complications for strided downloads just for this debug check
**/

void DownloadDump( const SCSGPrimSpec& d_ps )
{
     for(unsigned i=0 ; i < d_ps.num_prim ; i++)
     {
         const unsigned* u_ptr = d_ps.sbtIndexOffset + (d_ps.stride_in_bytes/sizeof(unsigned))*i ;
         const float*    f_ptr = d_ps.aabb           + (d_ps.stride_in_bytes/sizeof(float))*i ;

         float*     f = CU::DownloadArray<float>( f_ptr, 6 );
         unsigned*  u = CU::DownloadArray<unsigned>( u_ptr, 1 );

         std::cout << " off " << *(u) << " aabb (" << i << ") " ;
         for( unsigned i=0 ; i < 6 ; i++ ) std::cout << *(f+i) << " " ;
         std::cout << std::endl ;

         delete [] f ;
         delete [] u ;
     }
}

SCSGPrimSpec test_dspec( CSGPrim* d_prim , unsigned num)
{
     std::cout << "test_dspec" << std::endl ;
     SCSGPrimSpec d_ps = CSGPrim::MakeSpec( d_prim, 0, num );
     DownloadDump(d_ps);

     return d_ps ;
}


void test_pointer( const void* d, const char* label )
{
     std::cout << "test_pointer " << label << std::endl ;

     const void* vd = (const void*) d ;
     uintptr_t ud = (uintptr_t)d ; // uintptr_t is an unsigned integer type that is capable of storing a data pointer.
     CUdeviceptr cd = (CUdeviceptr) (uintptr_t) d ;  // CUdeviceptr is typedef to unsigned long lonh

     std::cout << "            (const void*) d " << vd << std::endl ;
     std::cout << "               (uintptr_t)d " << std::dec << ud << " " << std::hex << ud  << std::dec << std::endl ;
     std::cout << "  (CUdeviceptr)(uintptr_t)d " << std::dec << cd << " " << std::hex << cd  << std::dec << std::endl ;
}
#endif

int main(int argc, char** argv)
{
     test_AABB();
     test_offsets();

     std::vector<CSGPrim> prim ;
     for(unsigned i=0 ; i < 10 ; i++) prim.push_back(make_prim(float(i+1), i*10 ));

     test_spec(prim);
     test_partial(prim);

#ifdef WITH_CUDA
     CSGPrim* d_prim = test_upload(prim.data(), prim.size());
     test_download( d_prim, prim.size() );

     SCSGPrimSpec d_ps = test_dspec(d_prim, prim.size() ) ;

     test_pointer( d_prim, "d_prim" );
     test_pointer( d_ps.aabb ,           "d_ps.aabb" );
     test_pointer( d_ps.sbtIndexOffset , "d_ps.sbtIndexOffset" );
#endif

     return 0 ;
}

