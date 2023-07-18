// ./s_mock_texture_test.sh
/**



Aim is to be able to use qbnd.h in a mock cuda manner on CPU. 

**/

#include <cstdio>
#include "NPFold.h"
#include "s_mock_texture.h"



/**
conventional_size
--------------------

My convention for the size of an array of shape:: 

   (ni, nj, nk, nl, 4 )

* height : ni*nj*nk
* width  : nl 
* payload : 4 


**/

template<int P>
void conventional_size( int& width, int& height, const NP* a )
{
    assert(a); 
    const std::vector<int>& sh = a->shape ; 
    int nd = sh.size() ; 
    assert( nd > 1 && sh[nd-1] == P ); 
    width = sh[nd-2] ; 
    height = 1 ; 
    for(int i=0 ; i < nd-2 ; i++) height *= sh[i] ; 
}

cudaTextureObject_t create_boundary_texture()
{
    const char* base = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/standard" ;
    const NP* bnd     = NP::Load(base, "bnd.npy"); 

    int width, height ; 
    conventional_size<4>(width, height, bnd); 
 
    cudaTextureObject_t tex = MockTextureManager::Add(bnd, width, height) ;  ;  
    return tex ; 
}

NPFold* test_bnd()
{
    cudaTextureObject_t tex = create_boundary_texture(); 

    std::cout << MockTextureManager::Desc() ; 
    std::cout << " tex : " << tex << std::endl ; 

    MockTextureManager* mgr = MockTextureManager::Get();  
    std::cout << mgr->dump<float>(tex) ; 
    std::cout << mgr->dump<float4>(tex) ; 

    const NP* a = mgr->tt[tex].a ;
 
    int width, height ; 
    conventional_size<4>(width, height, a); 
 

    NP* b = NP::MakeLike(a) ; 
    float4* bb = b->values<float4>() ; 

    for(int iy=0 ; iy < height ; iy++)
    for(int ix=0 ; ix < width  ; ix++)
    {
        float x = (float(ix)+0.5f)/width   ; 
        float y = (float(iy)+0.5f)/height  ; 
        float4 payload = tex2D<float4>(tex, x, y ) ;
        int idx = iy*width + ix ; 
        bb[idx] = payload ; 
    }

    NPFold* f = new NPFold ; 
    f->add("a",   a ); 
    f->add("b",   b ); 
    return f ; 
}

NPFold* test_demo_3()
{
    int ny = 8 ; 
    int nx = 4 ; 
    int height = ny ; 
    int width = nx ; 

    NP* a0 = NP::Make<float>(ny,nx,4) ; 
    cudaTextureObject_t tex = MockTextureManager::Add(a0, width, height) ;  ;  
    MockTextureManager* mgr = MockTextureManager::Get();  

    NP* a = mgr->tt[tex].a ; 
    float4* aa = a->values<float4>() ; 

    for(int iy=0 ; iy < ny ; iy++ )
    for(int ix=0 ; ix < nx ; ix++ )
    {
        int idx = iy*nx+ix ; 
        aa[idx] = make_float4( iy, ix, 0.f, 0.f );   // slower dimension first 
    }

    NP* b = NP::MakeLike(a) ; 
    float4* bb = b->values<float4>() ; 

    for(int iy=0 ; iy < height ; iy++)
    for(int ix=0 ; ix < width  ; ix++)
    {
        float x = (float(ix)+0.5f)/width   ; 
        float y = (float(iy)+0.5f)/height  ; 
        int idx = iy*width + ix ; 
        bb[idx] = tex2D<float4>(tex, x, y ) ;
    }

    NPFold* f = new NPFold ; 
    f->add("a",   a ); 
    f->add("b",   b ); 
    return f ; 
}


NPFold* test_demo_5()
{
    int ni = 10 ; 
    int nj =  4 ; 
    int nk =  2 ; 
    int nl = 100 ; 
    int nn =  4 ; 

    int ny = ni*nj*nk ;  
    int nx = nl ;
 
    int height = ny ; 
    int width = nx ; 

    NP* a0 = NP::Make<float>(ni,nj,nk,nl,nn) ; 

    cudaTextureObject_t tex = MockTextureManager::Add(a0, width, height) ;  ;  
    MockTextureManager* mgr = MockTextureManager::Get();  

    NP* a = mgr->tt[tex].a ; 
    float4* aa = a->values<float4>() ; 

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    for(int k=0 ; k < nk ; k++)
    for(int l=0 ; l < nl ; l++)
    {
        int idx = i*nj*nk*nl + j*nk*nl + k*nl + l ; 
        aa[idx] = make_float4( i, j, k, l );   // slower dimension first 
    }

    NP* b = NP::MakeLike(a) ; 
    float4* bb = b->values<float4>() ; 

    for(int iy=0 ; iy < height ; iy++)
    for(int ix=0 ; ix < width  ; ix++)
    {
        float x = (float(ix)+0.5f)/width   ; 
        float y = (float(iy)+0.5f)/height  ; 
        int idx = iy*width + ix ; 
        bb[idx] = tex2D<float4>(tex, x, y ) ;
    }

    NPFold* f = new NPFold ; 
    f->add("a",   a ); 
    f->add("b",   b ); 
    return f ; 
}




/**
boundary_lookup
-----------------

HMM: this is duplicating stuff from qbnd. 
Thats not what is needed need to implement stuff that mocks what 
CUDA tex lookups are doing 
             
::


                  wl_samples
                  /
     (52, 4, 2, 761, 4, )
      |   |   \       \
     bnd omat paygrp   payvals
         osur
         isur
         imat 

    (ni, nj, nk, nl, nn )

    (bnd,species,paygrp) -> line "x"
    (wl) -> "y" 


See:: 

   QBnd::MakeBoundaryTex
   qbnd::boundary_lookup

**/

float4 boundary_lookup(cudaTextureObject_t obj,  float nm, int line, int k ) 
{
    MockTexture tex = MockTextureManager::Get(obj) ; 

    int nx, ny ; 
    conventional_size<4>(nx, ny, tex.a);   // HMM: could do this within MockTexture ?
 
    // follow qbnd::boundary_lookup
    float fx = (nm - tex.dom.x)/tex.dom.z ;   
    float x = (fx+0.5f)/float(nx) ; 

    int iy = 2*line + k ;    // k is 0 or 1 
    float y = (float(iy)+0.5f)/float(ny) ; 

    float4 props = tex2D<float4>(obj, x, y );  
    return props ; 
}


NPFold* test_boundary_lookup()
{
    cudaTextureObject_t obj = create_boundary_texture(); 
    MockTexture tex = MockTextureManager::Get(obj) ; 

    const std::vector<int>& sh = tex.a->shape ; 
    int nd = sh.size() ; 
    assert( nd == 5 ); 
    assert( sh[nd-1] == 4 );  

    int ni = sh[0] ;  // bnd     0... ~50
    int nj = sh[1] ;  // species 0,1,2,3
    int nk = sh[2] ;  // group   0,1 

    NP* a = NP::Make<float>(ni,nj,nk,4) ; 
    float4* aa = a->values<float4>();  

    float wavelength = 440.f ; 

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    for(int k=0 ; k < nk ; k++)
    {
        int line = i*nj+ j ; 
        float4 prop = boundary_lookup( obj, wavelength, line, k ); 

        int idx = i*nj*nk + j*nk + k ; 
        aa[idx] = prop ; 
    }

    NPFold* f = new NPFold ; 
    f->add("a", a) ; 
    return f ; 
}

int main(int argc, char** argv)
{
    //NPFold* f = test_demo_3(); 
    //NPFold* f = test_demo_5(); 
    //NPFold* f = test_bnd(); 
    NPFold* f = test_boundary_lookup(); 

    f->save("$FOLD"); 
    return 0;
}

