#include "G4MaterialPropertyVector.hh"
#include "X4MaterialPropertyVector.hh"
#include "NPY.hpp"

X4MaterialPropertyVector::X4MaterialPropertyVector(const G4MaterialPropertyVector* vec_ )
    :
    vec(vec_)
{
}

template <typename T> NPY<T>* X4MaterialPropertyVector::convert() 
{
    size_t num_val = vec->GetVectorLength() ; 
    NPY<T>* a = NPY<T>::make( num_val, 2 );  
    a->zero(); 
    int k=0 ; 
    int l=0 ; 
    for(size_t i=0 ; i < num_val ; i++)
    {   
        G4double energy = vec->Energy(i); 
        G4double value = (*vec)[i] ; 
        a->setValue(i, 0, k, l,  T(energy) );  
        a->setValue(i, 1, k, l,  T(value) );  
    }   
    return a ;   
}


template NPY<float>*  X4MaterialPropertyVector::convert() ; 
template NPY<double>* X4MaterialPropertyVector::convert() ; 




