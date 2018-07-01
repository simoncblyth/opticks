#pragma once

#include <vector>
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
    static T* Process( T* root_ , unsigned soIdx, unsigned lvIdx );
    static std::vector<unsigned>*  LVList ;  

    NTreeProcess(T* root_); 
    void init();

    T* root ; 
    T* balanced  ; 
    T* result  ; 

    NTreeBalance<T>*  balancer ; 
    NTreePositive<T>* positiver ; 

};



