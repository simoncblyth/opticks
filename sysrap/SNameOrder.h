#pragma once
/**
SNameOrder.h : Ordering vectors of objects with GetName methods
=================================================================

After x4/X4NameOrder.hh

**/

#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include "sstr.h"


template <typename T>
struct SNameOrder
{
    static void Sort( std::vector<T*>& a, bool reverse=false, const char* tail="0x" ); 
    static std::string Desc(const std::vector<T*>& a, const char* tail="0x", int w=50  );  

    SNameOrder(bool reverse_, const char* tail_);
    bool operator() (const T* a, const T* b) const  ;

    bool reverse ; 
    const char* tail ; 
};

template <typename T>
inline void SNameOrder<T>::Sort( std::vector<T*>& a, bool reverse, const char* tail ) // static
{
    SNameOrder<T> name_order(reverse, tail); 
    std::sort( a.begin(), a.end(), name_order );  
}

template <typename T>
inline std::string SNameOrder<T>::Desc(const std::vector<T*>& a, const char* tail, int w ) // static
{
    std::stringstream ss ; 
    ss << "SNameOrder::Desc" 
       << " tail " << ( tail ? tail : "-" )
       << std::endl 
       ; 
    for(unsigned i=0 ; i < a.size() ; i++)
    {
        T* obj = a[i] ;
        const std::string name = obj->GetName();
        const std::string sname = sstr::StripTail(name, tail ) ;
        ss
            << std::setw(4) << i
            << " : "
            << std::setw(w) << name
            << " : "
            << std::setw(w) << sname
            << std::endl ;
    }
    std::string str = ss.str(); 
    return str ; 
}

template <typename T>
inline SNameOrder<T>::SNameOrder(bool reverse_, const char* tail_) 
    :   
    reverse(reverse_),
    tail(tail_ ? strdup(tail_) : nullptr)
{
} 

template <typename T>
inline bool SNameOrder<T>::operator() (const T* a, const T* b) const
{
    std::string an = a->GetName();
    std::string bn = b->GetName();
    if(tail)
    {
        an = sstr::StripTail(an, tail);
        bn = sstr::StripTail(bn, tail);
    }
    bool cmp = an < bn ;
    return reverse ? !cmp : cmp  ;
}

