#include <cassert>
#include <cmath>
#include <iostream>

#include "SVec.hh"


template <typename T>
T SVec<T>::MaxDiff(const std::vector<T>& a, const std::vector<T>& b, bool dump)
{
    assert( a.size() == b.size() );
    T mx = 0.f ;     
    for(unsigned i=0 ; i < a.size() ; i++)
    {
        T df = std::abs(a[i] - b[i]) ; 
        if(df > mx) mx = df ; 

        if(dump)
        std::cout 
            << " a " << a[i] 
            << " b " << b[i] 
            << " df " << df
            << " mx " << mx 
            << std::endl 
            ; 

    }
    return mx ; 
}


template <typename T>
int SVec<T>::FindIndexOfValue(const std::vector<T>& a, T value, T tolerance)
{
    int idx = -1 ; 
    for(unsigned i=0 ; i < a.size() ; i++)
    {   
        T df = std::abs(a[i] - value) ; 
        if(df < tolerance)
        {   
            idx = i ; 
            break ; 
        }   
    }           
    return idx ; 
}
 



template struct SVec<float>;
template struct SVec<double>;


