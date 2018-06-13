#pragma once

#include "NPY_API_EXPORT.hh"
#include <vector>

/**
NNodeCollector
===============

Works with node structs with members: 

* left, right pointing to other nodes
* char* label  

Trying to stay const correct, makes difficult to 
use in NNodeAnalyse


**/

template <typename T>
struct NPY_API NNodeCollector
{
    NNodeCollector( T* root_ ); 
    ~NNodeCollector(); 

    void collect_preorder_r( T* node ); 
    void collect_inorder_r( T* node ); 
    void collect_postorder_r( T* node ); 

    void dump( const char* msg, std::vector<T*>& order ) ;
    void dump( const char* msg="NNodeCollector::dump" ) ;

    T* root ; 
    std::vector<T*> inorder ; 
    std::vector<T*> preorder ; 
    std::vector<T*> postorder ; 

};


