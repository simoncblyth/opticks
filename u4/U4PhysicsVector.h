#pragma once

#include "G4PhysicsVector.hh"
#include "G4PhysicsTable.hh"
#include "NP.hh"

struct U4PhysicsVector
{
    static NP* ConvertToArray(const G4PhysicsVector* prop) ;   
    static NP* CreateCombinedArray( const G4PhysicsTable* table ); 
}; 

/**

HMM same as U4MaterialPropertyVector::ConvertToArray( const G4MaterialPropertyVector* prop )

**/

inline NP* U4PhysicsVector::ConvertToArray(const G4PhysicsVector* prop) // static 
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

inline NP* U4PhysicsVector::CreateCombinedArray( const G4PhysicsTable* table )
{
    if(table == nullptr) return nullptr ; 
    if(table->size() == 0) return nullptr ; 
    std::vector<const NP*> aa ; 
    for(size_t i=0 ; i < table->size() ; i++)
    {
        G4PhysicsVector* vec = (*table)[i] ; 
        const NP* a = ConvertToArray(vec) ; 
        aa.push_back(a);  
    }
    return NP::Combine(aa); 
}





