#include <sstream>
#include <iomanip>


#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicsFreeVector.hh"
#include "G4PhysicsVector.hh"
#include "G4DataVector.hh"

#include "X4Array.hh"


#include "NPY.hpp"
#include "NP.hh"
#include "PLOG.hh"

const plog::Severity X4Array::LEVEL = PLOG::EnvLevel("X4Array", "DEBUG" ); 


X4Array* X4Array::Load(const char* base, const char* name, double en_scale )   // static 
{
    NPY<double>* a = NPY<double>::load(base, name); 
    if(en_scale > 0.) a->pscale(en_scale, 0u); 
    return X4Array::FromArray( a ); 
}

X4Array* X4Array::FromArray(const NPY<double>* a )   // static 
{
    assert( a && a->getNumDimensions() == 2 && a->hasShape(-1,2) && a->getNumItems() > 1 );  
    unsigned ni = a->getNumItems(); 
    const double* pdata = a->getValuesConst(); 
    X4Array* xvec = FromPairData( pdata, ni ); 
    xvec->src = a ; 
    return xvec ; 
}

X4Array* X4Array::FromArray(const NP* a )   // static 
{
    assert( a->is_pshaped() ); 
    unsigned ni = a->shape[0]; 
    const double* pdata = a->cvalues<double>(); 
    X4Array* xvec = FromPairData( pdata, ni ); 
    xvec->asrc = a ; 
    return xvec ; 
}

/**
X4Array::FromPairData
-----------------------

G4PhysicsVector bizarrely has no convenient ctor so use the G4PhysicsFreeVector ctor
and cast down to G4PhysicsVector using G4DataVector to fill that. 
 
**/

X4Array* X4Array::FromPairData(const double* pdata, unsigned npair )   // static 
{
    G4DataVector energy(npair, 0.) ;   // G4DataVector ISA std::vector<double>
    G4DataVector value(npair, 0.) ; 

    for(unsigned i=0 ; i < npair ; i++)
    {
        energy[i] = pdata[2*i+0]; 
        value[i] = pdata[2*i+1]; 
    }

    G4PhysicsFreeVector* pfv = new G4PhysicsFreeVector( energy, value ); 
    G4PhysicsVector* pv = static_cast<G4PhysicsVector*>( pfv ) ; 

    X4Array* xa = new X4Array(pv); 
    return xa ; 
}
 

template<typename T>
NPY<T>* X4Array::Convert(const G4PhysicsVector* vec )   // static 
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



std::string X4Array::Desc(const G4PhysicsVector* vec ) // static
{
    std::stringstream ss ; 
    
    size_t num_val = vec->GetVectorLength() ; 
    ss << "X4Array::Desc num_val " << num_val << std::endl ; 
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


X4Array::X4Array( G4PhysicsVector* vec_ )
    :
    src(nullptr),
    asrc(nullptr),
    vec(vec_)
{
}

std::string X4Array::desc() const 
{
    return Desc(vec); 
}

G4double X4Array::getMidBinValue() const 
{
    size_t num_val = vec->GetVectorLength() ; 
    size_t mid_bin = num_val/2 ; 
    return (*vec)[mid_bin] ; 
}


void X4Array::changeAllToMidBinValue()
{
    G4double midBinValue = getMidBinValue() ;
    putValues(midBinValue); 
}

void X4Array::putValues( G4double new_value )
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
NPY<T>* X4Array::convert()
{
    return Convert<T>(vec);  
}

template NPY<float>*  X4Array::convert() ; 
template NPY<double>* X4Array::convert() ; 

template NPY<float>*  X4Array::Convert(const G4PhysicsVector* vec) ; 
template NPY<double>* X4Array::Convert(const G4PhysicsVector* vec) ; 




