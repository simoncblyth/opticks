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
use in NTreeAnalyse


**/

template <typename T>
struct NPY_API NNodeCollector
{
    static void Inorder_r( std::vector<T*>& inorder, T* node ); 

    NNodeCollector( const T* root_ ); 
    ~NNodeCollector(); 

    void collect_preorder_r( const T* node ); 
    void collect_inorder_r( const T* node ); 
    void collect_postorder_r( const T* node ); 

    void dump( const char* msg, std::vector<const T*>& order ) ;
    void dump( const char* msg="NNodeCollector::dump" ) ;

    const T* root ; 
    std::vector<const T*> inorder ; 
    std::vector<const T*> preorder ; 
    std::vector<const T*> postorder ; 

};


