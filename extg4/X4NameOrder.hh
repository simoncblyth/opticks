#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

#include "SGDML.hh"

/**
X4NameOrder
---------------------------------

Comparator functor for sorting a vector of 
objects with GetName methods.  

When strip is true the name has any "0xc0ffee" removed before comparisons. 

**/

template <typename T>
struct X4NameOrder
{
    static void Dump(const char* msg, const std::vector<T*>& a );  

    X4NameOrder(bool reverse_, bool strip_);

    bool operator() (const T* a, const T* b) const  ;

    bool reverse ; 
    bool strip ; 
};

template <typename T>
X4NameOrder<T>::X4NameOrder(bool reverse_, bool strip_) 
    : 
    reverse(reverse_),
    strip(strip_)
{
} 

template <typename T>
void X4NameOrder<T>::Dump(const char* msg, const std::vector<T*>& a )
{
    std::cout << msg << std::endl ; 
    for(unsigned i=0 ; i < a.size() ; i++)
    {
        T* obj = a[i] ; 
        const std::string name = obj->GetName();
        const std::string sname = SGDML::Strip(name) ; 
        std::cout 
            << std::setw(4) << i 
            << " : " 
            << std::setw(80) << name 
            << " : " 
            << std::setw(80) << sname 
            << std::endl ; 
    }
}

template <typename T>
bool X4NameOrder<T>::operator() (const T* a, const T* b) const
{   
    std::string an = a->GetName();
    std::string bn = b->GetName();
    if(strip)
    {
        an = SGDML::Strip(an); 
        bn = SGDML::Strip(bn); 
    }
    bool cmp = an < bn ;  
    return reverse ? !cmp : cmp  ;
}   

