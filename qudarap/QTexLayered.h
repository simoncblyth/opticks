#pragma once
/**
QTexLayered.h
==============

Creates a CUDA layered texture from a NP.hh array of shape::

    (layers, height, width, payload )

Currently tested only with payload = 1 and T=float.

**/
#include <string>
#include <cstddef>

#include <cuda_runtime.h>
#include "cudaCheckErrors.h"
#include "QUDA_CHECK.h"

#include "scuda.h"
#include "squad.h"
#include "NP.hh"


template<typename T>
struct QTexLayered
{
    const NP*    a ;
    char         filterMode ;
    bool         scrunch_height ;

    size_t       layers ;
    size_t       height ;
    size_t       width ;
    size_t       payload ;

    int          hd_factor ;

    quad4*       meta ;
    quad4*       d_meta ;

    std::string desc() const ;
    QTexLayered(const NP* a, char filterMode, bool scrunch );
    ~QTexLayered();

    cudaChannelFormatDesc  channelDesc ;
    cudaArray_t            layeredArray;
    cudaTextureObject_t    tex ;

    static size_t GetPayloadSize();

    void init();
    void init_createArray();
    void init_createTextureObject();

    void uploadMeta();
};

template<typename T>
std::string QTexLayered<T>::desc() const
{
   std::stringstream ss ;
   ss << "QTexLayered"
      << "  a "  << ( a ? a->sstr() : "-" )
      << "  scrunch_height " << ( scrunch_height ? "YES" : "NO " )
      << "  layers " << layers
      << "  height " << height
      << "  width  " << width
      << "  payload " << payload
      << "  hd_factor " << hd_factor
      << "\n"
      ;

   std::string str = ss.str() ;
   return str ;
}



template<typename T>
QTexLayered<T>::QTexLayered(const NP* a_, char filterMode_, bool scrunch_height_ )
    :
    a(a_),
    filterMode(filterMode_),
    scrunch_height(scrunch_height_),
    layers(0),
    height(0),
    width(0),
    payload(0),
    hd_factor(a ? a->get_meta<int>("hd_factor",0) : -1 ),
    channelDesc(cudaCreateChannelDesc<T>()),
    layeredArray(nullptr),
    tex(0),
    meta(new quad4),
    d_meta(nullptr)
{
    init();
}

template<typename T>
QTexLayered<T>::~QTexLayered()
{
    cudaDestroyTextureObject(tex);
    cudaFreeArray(layeredArray);
}


template<typename T>
size_t QTexLayered<T>::GetPayloadSize() // static
{
    // assumes float based payload
    return sizeof(T)/sizeof(float);
}

template<typename T>
void QTexLayered<T>::init()
{
    assert(a);
    assert(a->shape.size() == 4 );
    assert( filterMode == 'P' || filterMode == 'L' );

    if(scrunch_height)
    {
        layers = a->shape[0]*a->shape[1];
        height = 1 ;
    }
    else
    {
        layers = a->shape[0];
        height = a->shape[1];
    }
    width  = a->shape[2];
    payload = a->shape[3] ;


    size_t payload_check = GetPayloadSize();
    assert( payload_check == payload );
    assert( payload == 1 || payload == 2 || payload == 3 || payload == 4 );

    meta->q0.u.x = width ;
    meta->q0.u.y = height ;
    meta->q0.u.z = layers ;   // HMM: payload can be determined from template type
    meta->q0.u.w = hd_factor ;

    init_createArray();
    init_createTextureObject();
}


template<typename T>
void QTexLayered<T>::init_createArray()
{
    // CUDA documentation for make_cudaExtent says that the first "width" arg
    // should be in bytes... Gemini assures me that is only the case for linear
    // memory, width in elements is needed for extent interpreted by cudaMalloc3DArray
    // which is needed for layered texture as the channelDesc carries the element size

    cudaMalloc3DArray(&layeredArray,
                      &channelDesc,
                      make_cudaExtent(width, height, layers),
                      cudaArrayLayered);

    float* host_data = (float*)a->bytes();

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(
          host_data,                 // Pointer to start of memory
          width * sizeof(float),     // PITCH: bytes from Row N to Row N+1 (distance between rows in same layer)
          width,                     // logical width
          height);                   // logical height
    copyParams.dstArray = layeredArray;
    copyParams.extent = make_cudaExtent(width, height, layers);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
}

template<typename T>
void QTexLayered<T>::init_createTextureObject()
{
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = layeredArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;


    switch(filterMode)
    {
        case 'L': texDesc.filterMode = cudaFilterModeLinear ; break ;  // smooth linear interpolation
        case 'P': texDesc.filterMode = cudaFilterModePoint  ; break ;  // no-interpolation, eg for lookup testing
        // cudaFilterModePoint: switches off interpolation, necessary with char texture
    }

    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;                // Use 0.0 -> 1.0 for probability lookup

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
}


/**
QTexLayered<T>::uploadMeta
---------------------------

This is not called from init to allow user to add metadata
before upload.

**/

template<typename T>
void QTexLayered<T>::uploadMeta()
{
    size_t size = sizeof(quad);
    d_meta = nullptr ;
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_meta ), size ));
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_meta ), meta, size, cudaMemcpyHostToDevice ));
}

