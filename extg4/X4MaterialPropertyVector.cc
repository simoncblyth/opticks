#include "G4MaterialPropertyVector.hh"
#include "X4MaterialPropertyVector.hh"
#include "NPY.hpp"


G4MaterialPropertyVector* X4MaterialPropertyVector::FromArray(const NPY<double>* arr)
{
    size_t ni = arr->getShape(0); 
    size_t nj = arr->getShape(1); 
    assert( nj == 2 ); 

    G4double* energy = new G4double[ni] ;
    G4double* value = new G4double[ni] ;
 
    for(int i=0 ; i < int(ni) ; i++)
    {
        energy[i] = arr->getValue(i, 0, 0 ); 
        value[i] = arr->getValue(i, 1, 0); 
    }

    G4MaterialPropertyVector* vec = new G4MaterialPropertyVector(energy, value, ni);  
    return vec ; 
}



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




