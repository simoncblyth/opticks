#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "SYSRAP_API_EXPORT.hh"

/**
SNameVec<T>
=============

Sorts a vector of T* where the T has a 
GetName method returning something that can be 
converted into std::string.  


**/
 
template <typename T>
struct SYSRAP_API SNameVec
{
    static void Dump(const std::vector<T*>& a );    
    static void Sort(      std::vector<T*>& a, bool reverse );    
};

template <typename T>
struct SYSRAP_API SNameVecOrder
{
    SNameVecOrder(bool reverse_) : reverse(reverse_) {} 

    bool operator() (const T* a, const T* b) const 
    {   
        const std::string an = a->GetName() ; 
        const std::string bn = b->GetName() ; 
        bool cmp = an < bn ; 
        return reverse ? !cmp : cmp ;
    }   
    bool reverse ; 
};


template <typename T>
void SNameVec<T>::Dump( const std::vector<T*>& a  )
{
    for(unsigned i=0 ; i < a.size() ; i++) std::cout << std::setw(10) << a[i]->GetName() << std::endl  ; 
} 

template <typename T>
void SNameVec<T>::Sort( std::vector<T*>& a, bool reverse)
{
    SNameVecOrder<T> order(reverse) ; 
    std::sort( a.begin(),  a.end(),  order ); 
} 


