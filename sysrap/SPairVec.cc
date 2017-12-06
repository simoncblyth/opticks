#include <algorithm>
#include <iostream>

#include "SPairVec.hh"
#include "PLOG.hh"


template <typename K, typename V>
SPairVec<K,V>::SPairVec( LPKV& lpkv, bool ascending )
   :
   _lpkv(lpkv), 
   _ascending(ascending)
{
}

template <typename K, typename V>
bool SPairVec<K,V>::operator()(const PKV& a, const PKV& b)
{
    return _ascending ? a.second < b.second : a.second > b.second ; 
}

template <typename K, typename V>
void SPairVec<K,V>::sort()
{
    std::sort( _lpkv.begin(), _lpkv.end(), *this );
}


template <typename K, typename V>
void SPairVec<K,V>::dump(const char* msg) const 
{
    LOG(info) << msg << " size " << _lpkv.size() ; 
    for(unsigned i=0 ; i < _lpkv.size() ; i++)
    {
        std::cerr 
            << " " << _lpkv[i].first 
            << " " << _lpkv[i].second
            << std::endl 
            ;
    }
}



template struct SPairVec<std::string, unsigned>;


