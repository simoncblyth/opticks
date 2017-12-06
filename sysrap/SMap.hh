#pragma once

#include <map>
#include <vector>

#include "SYSRAP_API_EXPORT.hh"
 
template <typename K, typename V>
struct SYSRAP_API SMap
{
    typedef typename std::vector<K> KK ; 
    typedef typename std::map<K,V> MKV ; 

    static unsigned ValueCount(const MKV& m, V value ); 

    static void FindKeys(const MKV& m, KK& keys, V value, bool dump ); 

};


