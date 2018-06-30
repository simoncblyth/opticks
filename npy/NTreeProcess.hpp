#pragma once

#include "NPY_API_EXPORT.hh"

/**
NTreeProcess
==============

**/

template <typename T> class  NTreePositive ; 
template <typename T> struct NTreeBalance ; 

template <typename T>
struct NPY_API NTreeProcess
{
    static unsigned MaxHeight0 ;  
    NTreeProcess(T* root_); 
    void init();


    T* root ; 
    T* balanced  ; 
    T* result  ; 

    NTreeBalance<T>*  balancer ; 
    NTreePositive<T>* positiver ; 

};



