#pragma once
/**
U4MaterialPropertyVector.h
============================


After X4MaterialPropertyVector.hh

**/

#include "G4MaterialPropertyVector.hh"
#include "NP.hh"

struct U4MaterialPropertyVector
{
    static G4MaterialPropertyVector* FromArray(const NP* prop);
    static NP* ConvertToArray(const G4MaterialPropertyVector* vec);
};





inline NP* U4MaterialPropertyVector::ConvertToArray(const G4MaterialPropertyVector* prop)
{
    size_t num_val = prop->GetVectorLength() ; 
    NP* a = NP::Make<double>( num_val, 2 );  
    double* a_v = a->values<double>(); 
    for(size_t i=0 ; i < num_val ; i++)
    {   
        G4double energy = prop->Energy(i); 
        G4double value = (*prop)[i] ;
        a_v[2*i+0] = energy ; 
        a_v[2*i+1] = value ; 
    }   
    return a ;   
}


G4MaterialPropertyVector* U4MaterialPropertyVector::FromArray(const NP* a ) // static 
{   
    assert( a->uifc == 'f' && a->ebyte == 8 );
    
    size_t ni = a->shape[0] ;
    size_t nj = a->shape[1] ;
    assert( nj == 2 );
    
    G4double* energy = new G4double[ni] ;
    G4double* value = new G4double[ni] ;
    
    for(int i=0 ; i < int(ni) ; i++)
    {   
        energy[i] = a->get<double>(i,0) ;
        value[i] = a->get<double>(i,1) ;
    }
    G4MaterialPropertyVector* vec = new G4MaterialPropertyVector(energy, value, ni);
    return vec ;
}



