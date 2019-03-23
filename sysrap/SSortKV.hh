#pragma once

/**
SSortKV
==========

Utility struct for reordering a vector of string float pairs.

**/


#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include "SYSRAP_API_EXPORT.hh"
 
struct SYSRAP_API SSortKV 
{
    typedef std::pair<std::string, float> KV ; 
    typedef std::vector<KV> VKV ;

    SSortKV(bool descending_) : descending(descending_) {} 

    void add(const char* k, float v)
    {    
        vkv.push_back(KV(k, v));
    }    
    void sort()
    {    
        std::sort(vkv.begin(), vkv.end(), *this );
    }    
    void dump(const char* msg="SSortKV::dump") const ;

    bool operator()( const KV& a, const KV& b) const 
    {    
        return descending ? a.second > b.second : a.second < b.second ; 
    }    
    unsigned getNum() const  
    {
        return vkv.size();
    }
    const std::string& getKey(unsigned i) const 
    {    
        return vkv[i].first ; 
    }    
    float getVal(unsigned i) const 
    {    
        return vkv[i].second ; 
    }    

    VKV   vkv ; 
    bool  descending ;  

};

