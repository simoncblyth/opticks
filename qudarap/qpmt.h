#pragma once
/**
qpmt.h
=======


**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QPMT_METHOD __device__
#else
   #define QPMT_METHOD 
#endif 


template <typename T> struct qprop ;

template<typename T>
struct qpmt
{
    qprop<T>* rindex ;
    T*        thickness ; 
}; 



