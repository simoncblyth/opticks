#include <cassert>
#include <cmath>
#include <iostream>

#include "SMap.hh"
#include "PLOG.hh"


template <typename K, typename V>
unsigned SMap<K,V>::ValueCount(const std::map<K,V>& m, V value)
{
    unsigned count(0); 
    for(typename MKV::const_iterator it=m.begin() ; it != m.end() ; it++)
    {
        V v = it->second ; 
        if( v == value ) count++ ; 
    }
    return count ; 
}


template <typename K, typename V>
void SMap<K,V>::FindKeys(const std::map<K,V>& m, std::vector<K>& keys, V value, bool dump)
{
    if(dump)
    {
        LOG(info) << " value " << std::setw(32) << std::hex << value << std::dec ; 
    } 

    for(typename MKV::const_iterator it=m.begin() ; it != m.end() ; it++)
    {
        K k = it->first ; 
        V v = it->second ; 
        bool match = v == value ; 

        if(dump)
        {
            LOG(info) 
                 << " k " << k 
                 << " v " << std::setw(32) << std::hex << v << std::dec 
                 << " match " << ( match ? "Y" : "N" )
                 << " keys.size() " << keys.size()
                 ;
        } 

        if( match ) keys.push_back(k) ; 
    }
}







template struct SMap<std::string, unsigned long long>;


