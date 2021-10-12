#pragma once

#include "QUDARAP_API_EXPORT.hh"
template <typename T> struct QTex ; 

template<typename T>
struct QUDARAP_API QTexRotate
{
    const QTex<T>* tex ; 

    T*           rotate_dst ; 
    T*           d_rotate_dst ; 

    QTexRotate( const QTex<T>* tex_ ) ; 
    virtual ~QTexRotate();  

    void rotate(float theta);
};



