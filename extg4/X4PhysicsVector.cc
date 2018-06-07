#include "GProperty.hh"

#include "G4PhysicsVector.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "X4PhysicsVector.hh"
#include "PLOG.hh"


template <typename T>
GProperty<T>* X4PhysicsVector<T>::Convert(const G4PhysicsVector* vec ) 
{
    X4PhysicsVector<T> xvec(vec); 
    GProperty<T>* prop = xvec.getProperty();
    return prop ; 
}

template <typename T>
X4PhysicsVector<T>::X4PhysicsVector( const G4PhysicsVector* vec )
    :
    m_vec(vec),
    m_prop(NULL)
{
    init();
}

template <typename T>
GProperty<T>* X4PhysicsVector<T>::getProperty() const
{
    return m_prop ; 
}

template <typename T>
void X4PhysicsVector<T>::init()
{
    size_t len = getVectorLength() ; 

    // converting assumed ascending energies domain to ascending wavelengths 
    // so must reverse ordering
    bool reverse = true ;   
    T* values  = getValues(reverse); 
    T* domain = getWavelengths(reverse); 

    m_prop = new GProperty<T>( values, domain, len );   

    // GProperty ctor copies inputs, so cleanup 
    delete[] values ;
    delete[] domain ;
}

template <typename T>
size_t X4PhysicsVector<T>::getVectorLength() const 
{
    return m_vec->GetVectorLength() ; 
}
 
template <typename T>
T* X4PhysicsVector<T>::getValues(bool reverse) const
{
    size_t n = getVectorLength() ; 
    T* a = new T[n] ; 
    for (size_t i=0; i<n; i++) a[reverse ? n-1-i : i] = (*m_vec)[i] ; 
    return a ; 
}

template <typename T>
T* X4PhysicsVector<T>::getEnergies(bool reverse) const
{
    size_t n = getVectorLength() ; 
    T* a = new T[n] ; 
    for (size_t i=0; i<n; i++) a[reverse ? n-1-i : i ] = m_vec->Energy(i) ; 
    return a ; 
}

template <typename T>
T* X4PhysicsVector<T>::getWavelengths(bool reverse) const 
{
    size_t n = getVectorLength() ; 
    T* a = new T[n] ; 

    double hc = h_Planck*c_light/(nm*eV) ;  // ~1240 nm.eV 

    for (size_t i=0; i<n; i++) 
    {
        T energy = m_vec->Energy(i)/eV ;  // convert into eV   (assumes input in G4 standard energy unit (MeV))
        T wavelength = hc/energy ;        // wavelength in nm

        a[reverse ? n-1-i : i] = wavelength ; 
    }
    return a ; 
}

template class X4PhysicsVector<float>;
template class X4PhysicsVector<double>;

