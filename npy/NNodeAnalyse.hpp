#pragma once

#include "NPY_API_EXPORT.hh"

/**
NNodeAnalyse
==============

* see ../analytic/csg.py 

**/

#include <string>
template <typename T> struct NNodeCollector ; 
template <typename T> struct NGrid ; 

template <typename T>
struct NPY_API NNodeAnalyse
{
    NNodeAnalyse(T* root_); 
    ~NNodeAnalyse(); 

    void init(); 
    void initGrid(); 
    unsigned depth_(bool label);
    unsigned depth_r(T* node, unsigned depth, bool label);
    std::string desc() const ;

    T*                 root ; 
    unsigned           height ; 
    NNodeCollector<T>* nodes ; 
    unsigned           count ; 
    NGrid<T>*          grid ; 
 

};



