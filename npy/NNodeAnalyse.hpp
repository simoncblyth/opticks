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
    static std::string Desc(const T* root_);
  
    NNodeAnalyse(const T* root_); 
    ~NNodeAnalyse(); 

    void init(); 
    void initGrid(); 
    unsigned depth_(bool label);
    unsigned depth_r(const T* node, unsigned depth, bool label);
    std::string desc() const ;

    const T*           root ; 
    unsigned           height ; 
    NNodeCollector<T>* nodes ; 
    unsigned           count ; 
    NGrid<T>*          grid ; 
 

};



