/**
s_mock_texture_test.cc
========================

Aim is to be able to use QBnd.hh/qbnd.h in MOCK_CUDA manner on CPU. 

::

   ./s_mock_texture_test.sh


**/

#include <cstdio>
#include "NPFold.h"

#include "s_mock_texture.h"
MockTextureManager* MockTextureManager::INSTANCE = nullptr ; 


cudaTextureObject_t create_boundary_texture()
{
    const char* base = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/standard" ;
    const NP* bnd     = NP::Load(base, "bnd.npy"); 


    cudaTextureObject_t tex = MockTextureManager::Add(bnd) ;  ;  
    return tex ; 
}

NPFold* test_bnd()
{
    cudaTextureObject_t obj = create_boundary_texture(); 

    std::cout << MockTextureManager::Desc() ; 
    std::cout << " obj : " << obj << std::endl ; 

    MockTextureManager* mgr = MockTextureManager::Get();  
    std::cout << mgr->dump<float>(obj) ; 
    std::cout << mgr->dump<float4>(obj) ; 

    const MockTexture tex = mgr->get(obj) ;  
    const NP* a = tex.a ;
 

    NP* b = NP::MakeLike(a) ; 
    float4* bb = b->values<float4>() ; 

    for(int iy=0 ; iy < tex.height ; iy++)
    for(int ix=0 ; ix < tex.width  ; ix++)
    {
        float x = (float(ix)+0.5f)/tex.width   ; 
        float y = (float(iy)+0.5f)/tex.height  ; 
        float4 payload = tex2D<float4>(obj, x, y ) ;
        int idx = iy*tex.width + ix ; 
        bb[idx] = payload ; 
    }

    NPFold* f = new NPFold ; 
    f->add("a",   a ); 
    f->add("b",   b ); 
    return f ; 
}

NPFold* test_demo_3()
{
    int ny = 8 ;  // height
    int nx = 4 ;  // width 

    NP* a0 = NP::Make<float>(ny,nx,4) ; 
    cudaTextureObject_t obj = MockTextureManager::Add(a0) ;  ;  
    MockTextureManager* mgr = MockTextureManager::Get();  

    NP* a = mgr->tt[obj].a ; 
    float4* aa = a->values<float4>() ; 

    for(int iy=0 ; iy < ny ; iy++ )
    for(int ix=0 ; ix < nx ; ix++ )
    {
        int idx = iy*nx+ix ; 
        aa[idx] = make_float4( iy, ix, 0.f, 0.f );   // slower dimension first 
    }

    NP* b = NP::MakeLike(a) ; 
    float4* bb = b->values<float4>() ; 

    for(int iy=0 ; iy < ny ; iy++)
    for(int ix=0 ; ix < nx  ; ix++)
    {
        float x = (float(ix)+0.5f)/nx   ; 
        float y = (float(iy)+0.5f)/ny  ; 
        int idx = iy*nx + ix ; 
        bb[idx] = tex2D<float4>(obj, x, y ) ;
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

    cudaTextureObject_t obj = MockTextureManager::Add(a0) ;  ;  
    MockTextureManager* mgr = MockTextureManager::Get();  

    MockTexture tex = mgr->get(obj) ; 

    NP* a = const_cast<NP*>(tex.a) ;  // unusual to set the values after adding 
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

    for(int iy=0 ; iy < tex.height ; iy++)
    for(int ix=0 ; ix < tex.width  ; ix++)
    {
        float x = (float(ix)+0.5f)/tex.width   ; 
        float y = (float(iy)+0.5f)/tex.height  ; 
        int idx = iy*tex.width + ix ; 
        bb[idx] = tex2D<float4>(obj, x, y ) ;
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
 
    // follow qbnd::boundary_lookup
    float fx = (nm - tex.dom.x)/tex.dom.z ;   
    float x = (fx+0.5f)/float(tex.width) ; 

    int iy = 2*line + k ;    // k is 0 or 1 
    float y = (float(iy)+0.5f)/float(tex.height) ; 

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
    NPFold* f = test_bnd(); 
    //NPFold* f = test_boundary_lookup(); 

    f->save("$FOLD"); 
    return 0;
}

