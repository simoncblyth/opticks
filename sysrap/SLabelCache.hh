#pragma once

/**
SLabelCache
===============

**/

#include "SYSRAP_API_EXPORT.hh"
#include <unordered_map>

template <typename T>
struct SYSRAP_API SLabelCache
{
    typedef std::unordered_map<const void*, T>  MVT ; 

    T missing ; 
    MVT cache ; 

    SLabelCache( T missing );

    void add(const void* obj, T label); 
    T find(const void* obj) ; 
}; 


template <typename T>
SLabelCache<T>::SLabelCache(T missing_) : missing(missing_) {}

template <typename T>
inline void SLabelCache<T>::add(const void* obj, T label)
{
    cache[obj] = label ; 
}
template <typename T>
inline T SLabelCache<T>::find(const void* obj)
{
    typename MVT::const_iterator en = cache.end();  
    typename MVT::const_iterator it = cache.find( obj );
    return it == en ? missing : it->second ; 
}


