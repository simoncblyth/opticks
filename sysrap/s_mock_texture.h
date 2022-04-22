#pragma once

/**
s_mock_texture : exploring CUDA texture lookup API on CPU
=============================================================

TODO:

* MockTextureManager needs API to collect cudaTextureObject_t indices and 
  associated NP array pointers to be used for the lookup

* NP arrays need to be the same ones uploaded to create the actual textures 

**/

#include <vector_types.h>
#include "scuda.h"

struct MockTextureManager 
{
    static MockTextureManager* INSTANCE ; 
    MockTextureManager() ;
    template<typename T> T tex2D(cudaTextureObject_t tex, float x, float y ); 
};

MockTextureManager* MockTextureManager::INSTANCE = nullptr ; 

inline MockTextureManager::MockTextureManager()
{
    INSTANCE = this ; 
}

template<> float MockTextureManager::tex2D<float>( cudaTextureObject_t tex, float x, float y )
{
    return 0.f ; 
}

template<> float4 MockTextureManager::tex2D<float4>( cudaTextureObject_t tex, float x, float y )
{
    return make_float4(0.f, 0.f, 0.f, 0.f)  ; 
}

template<typename T> T tex2D(cudaTextureObject_t tex, float x, float y )
{
    printf("//tex2d instance %p \n", MockTextureManager::INSTANCE ); 
    return MockTextureManager::INSTANCE->tex2D<T>( tex, x, y ); 
}




