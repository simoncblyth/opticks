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



template struct SVec<float>;
template struct SVec<double>;
