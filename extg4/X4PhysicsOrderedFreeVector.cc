#include <sstream>
#include <iomanip>


#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicsOrderedFreeVector.hh"

#include "X4PhysicsOrderedFreeVector.hh"


#include "NPY.hpp"
#include "PLOG.hh"

const plog::Severity X4PhysicsOrderedFreeVector::LEVEL = PLOG::EnvLevel("X4PhysicsOrderedFreeVector", "DEBUG" ); 


template<typename T>
NPY<T>* X4PhysicsOrderedFreeVector::Convert(const G4PhysicsOrderedFreeVector* vec )   // static 
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



std::string X4PhysicsOrderedFreeVector::Desc(const G4PhysicsOrderedFreeVector* vec ) // static
{
    std::stringstream ss ; 
    
    size_t num_val = vec->GetVectorLength() ; 
    ss << "X4PhysicsOrderedFreeVector::Desc num_val " << num_val << std::endl ; 
    for(size_t i=0 ; i < num_val ; i++)
    {
        G4double energy = vec->Energy(i); 
        G4double wavelength = h_Planck*c_light/energy ;
        G4double value  = (*vec)[i];
        ss 
            << "    " << std::setw(3) << i 
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << wavelength/nm  << " nm "
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << energy/eV << " eV "
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << value/mm << " mm "
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << value/m << " m  "
            << std::endl 
            ; 
    }
    std::string str = ss.str(); 
    return str ; 
}


X4PhysicsOrderedFreeVector::X4PhysicsOrderedFreeVector( G4PhysicsOrderedFreeVector* vec_ )
    :
    vec(vec_)
{
}

std::string X4PhysicsOrderedFreeVector::desc() const 
{
    return Desc(vec); 
}

G4double X4PhysicsOrderedFreeVector::getMidBinValue() const 
{
    size_t num_val = vec->GetVectorLength() ; 
    size_t mid_bin = num_val/2 ; 
    return (*vec)[mid_bin] ; 
}


void X4PhysicsOrderedFreeVector::changeAllToMidBinValue()
{
    G4double midBinValue = getMidBinValue() ;
    putValues(midBinValue); 
}

void X4PhysicsOrderedFreeVector::putValues( G4double new_value )
{
    size_t num_val = vec->GetVectorLength() ; 
    for(size_t i=0 ; i < num_val ; i++)
    {
        G4double old_value  = (*vec)[i];
        
        vec->PutValue(i, new_value ); 
        LOG(LEVEL)
            << " old_value " << std::setw(10) << std::setprecision(3) << std::fixed << old_value 
            << " new_value " << std::setw(10) << std::setprecision(3) << std::fixed << new_value 
            ;
    }
}



template<typename T>
NPY<T>* X4PhysicsOrderedFreeVector::convert()
{
    return Convert<T>(vec);  
}

template NPY<float>*  X4PhysicsOrderedFreeVector::convert() ; 
template NPY<double>* X4PhysicsOrderedFreeVector::convert() ; 

template NPY<float>*  X4PhysicsOrderedFreeVector::Convert(const G4PhysicsOrderedFreeVector* vec) ; 
template NPY<double>* X4PhysicsOrderedFreeVector::Convert(const G4PhysicsOrderedFreeVector* vec) ; 




