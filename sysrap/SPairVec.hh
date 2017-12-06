#pragma once

#include <map>
#include <vector>

#include "SYSRAP_API_EXPORT.hh"
 
template <typename K, typename V>
struct SYSRAP_API SPairVec
{
    typedef typename std::pair<K,V>    PKV ; 
    typedef typename std::vector<PKV>  LPKV ; 

    SPairVec( LPKV& lpkv, bool ascending ); 

    bool operator()(const PKV& a, const PKV& b);
    void sort(); 
    void dump(const char* msg) const ; 

    LPKV&  _lpkv ; 
    bool   _ascending ; 

};


