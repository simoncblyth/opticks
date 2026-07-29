#pragma once
/**
qplanck.h
==========

Instanciation and uploading of qplanck.h is orchestrated within
the QPlanck::QPlanck ctor which prepares it with QPlanck::MakeInstance
and uploads.


**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QPLANCK_METHOD __device__
#else
   #define QPLANCK_METHOD
#endif


struct qplanck
{
    cudaTextureObject_t tex ;

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
    QPLANCK_METHOD float wavelength(const float u0) const ;
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)

inline QPLANCK_METHOD float qplanck::wavelength(const float u0) const
{
    constexpr float y0 = 0.5f ;
    return tex2D<float>(tex, u0, y0 );
}

#endif

