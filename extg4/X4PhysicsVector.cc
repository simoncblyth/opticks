/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include "G4PhysicsVector.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "X4PhysicsVector.hh"

#include "GDomain.hh"
#include "GProperty.hh"
#include "SDigest.hh"
#include "SDirect.hh"
#include "PLOG.hh"


template <typename T>
const plog::Severity X4PhysicsVector<T>::LEVEL = PLOG::EnvLevel("X4PhysicsVector", "DEBUG" ) ; 


template <typename T>
std::string X4PhysicsVector<T>::Digest(const G4PhysicsVector* vec)  // see cfg4.G4PhysicsOrderedFreeVectorTest
{   
    if(!vec) return "" ;
    bool nm_domain = true ; 
    GProperty<T>* prop = Convert( vec, nm_domain ) ; 
    return prop->getDigestString(); 
}

/**
Digest0 is reproducibly SIGABRT crashing with G4 10.4.2  on macOS High Sierra
**/

template <typename T>
std::string X4PhysicsVector<T>::Digest0(const G4PhysicsVector* vec)  // see cfg4.G4PhysicsOrderedFreeVectorTest
{   
    if(!vec) return "" ;

    std::ofstream fp("/dev/null", std::ios::out);
    std::stringstream ss ;
    stream_redirect rdir(ss,fp); // stream_redirect such that writes to the file instead go to the stringstream 

    const_cast<G4PhysicsVector*>(vec)->Store(fp, false );

    std::string s = ss.str();  // std::string can hold \0  (ie that is not interpreted as null terminator) so they can hold any binary data 

    SDigest dig ;
    dig.update( const_cast<char*>(s.data()), s.size() );

    return dig.finalize();
}


template <typename T>
GProperty<T>* X4PhysicsVector<T>::Convert(const G4PhysicsVector* vec, bool nm_domain ) 
{
    X4PhysicsVector<T> xvec(vec, nullptr, nm_domain); 
    GProperty<T>* prop = xvec.getProperty();
    return prop ; 
}

template <typename T>
GProperty<T>* X4PhysicsVector<T>::Interpolate(const G4PhysicsVector* vec, const GDomain<T>* dom) 
{
    bool nm_domain = true ; 
    X4PhysicsVector<T> xvec(vec, dom, nm_domain ); 
    GProperty<T>* prop = xvec.getProperty();
    return prop ; 
}







template <typename T>
GProperty<T>* X4PhysicsVector<T>::getProperty() const
{
    return m_prop ; 
}

template <typename T>
X4PhysicsVector<T>::X4PhysicsVector( const G4PhysicsVector* vec, const GDomain<T>* dom, bool nm_domain )
    :
    m_vec(vec),
    m_dom(dom),
    m_prop(m_dom == nullptr ? makeDirect(nm_domain) : makeInterpolated())
{
}


/**
X4PhysicsVector::makeDirect
-----------------------------

Converting assumed ascending energies domain to ascending wavelengths 
so must reverse ordering.

**/

template <typename T>
GProperty<T>* X4PhysicsVector<T>::makeDirect(bool nm_domain) const
{
    bool reverse = nm_domain ;   
    size_t len = getSrcVectorLength() ; 
    LOG(LEVEL) << " src.len " << len ; 
    T* values = getSrcValues(reverse); 
    T* domain = nm_domain ? getSrcWavelengths(reverse) : getSrcEnergies(reverse) ; 
    GProperty<T>* prop = new GProperty<T>( values, domain, len );   

    // GProperty ctor copies inputs, so cleanup 
    delete[] values ;
    delete[] domain ;

    return prop ; 
}

template <typename T>
GProperty<T>* X4PhysicsVector<T>::makeInterpolated() const
{
    size_t len = m_dom->getLength() ; 
    LOG(LEVEL) << " dom.len " << len ; 

    T* domain = m_dom->getValues() ; 
    T* values = getInterpolatedValues(domain, len); 

    GProperty<T>* prop = new GProperty<T>( values, domain, len );   

    // GProperty ctor copies inputs, so cleanup 
    delete[] values ;
    delete[] domain ;

    return prop ; 
}


template <typename T>
T X4PhysicsVector<T>::_hc_eVnm()  // 1239.84...
{
    return h_Planck*c_light/(eV*nm) ; 
}


template <typename T>
T* X4PhysicsVector<T>::getInterpolatedValues(T* wavelength_nm, size_t n, T hc_eVnm_ ) const
{
    T* a = new T[n] ; 

    T hc_eVnm = hc_eVnm_ > 1239. && hc_eVnm_ < 1241. ? hc_eVnm_ : _hc_eVnm()  ; 

    for (size_t i=0; i<n; i++) 
    {
        T wl_nm = wavelength_nm[i] ; 
        T en_eV = hc_eVnm/wl_nm ; 
        T value = m_vec->Value(en_eV*eV); 
        a[i] = value ;

        /*
        std::cout
            << " i " << std::setw(4) << i 
            << " wl_nm " << std::setw(10) << std::fixed << std::setprecision(3) << wl_nm
            << " en_eV " << std::setw(10) << std::fixed << std::setprecision(3) << en_eV
            << " value " << std::setw(10) << value 
            << " hc_eVnm " << std::setw(10) << std::fixed << std::setprecision(3) << hc_eVnm 
            << " _hc_eVnm " << std::setw(10) << std::fixed << std::setprecision(3) << _hc_eVnm() 
            << std::endl 
            ;  
        */
    }
    return a ; 
}



template <typename T>
size_t X4PhysicsVector<T>::getSrcVectorLength() const 
{
    return m_vec->GetVectorLength() ; 
}
 

template <typename T>
T* X4PhysicsVector<T>::getSrcValues(bool reverse) const
{
    size_t n = getSrcVectorLength() ; 
    T* a = new T[n] ; 
    for (size_t i=0; i<n; i++) 
    {
        T value = (*m_vec)[i] ; 
        a[reverse ? n-1-i : i] = value ;
        LOG(LEVEL) 
            << " i " << std::setw(4) << i 
            << " value " << std::setw(10) << value 
            ;  
    }
    return a ; 
}

/**
X4PhysicsVector::getSrcEnergies
--------------------------------

Allocate array of T and fills asis from source vector.

**/

template <typename T>
T* X4PhysicsVector<T>::getSrcEnergies(bool reverse) const
{
    size_t n = getSrcVectorLength() ; 
    T* a = new T[n] ; 
    for (size_t i=0; i<n; i++) 
    {
        T energy = m_vec->Energy(i) ;
        a[reverse ? n-1-i : i ] = energy ; 

        LOG(LEVEL) 
            << " i " << std::setw(4) << i 
            << " energy " << std::setw(10) << energy   
            ;

    }
    return a ; 
}

/**
X4PhysicsVector::getSrcWavelengths
-------------------------------------

Allocate array of T and fills asis from source vector.

**/


template <typename T>
T* X4PhysicsVector<T>::getSrcWavelengths(bool reverse) const 
{
    size_t n = getSrcVectorLength() ; 
    T* a = new T[n] ; 

    double hc_eVnm = h_Planck*c_light/(nm*eV) ;  // ~1240 nm.eV 

    for (size_t i=0; i<n; i++) 
    {
        T energy_eV = m_vec->Energy(i)/eV ;     // convert into eV   (assumes input in G4 standard energy unit (MeV))
        T wavelength_nm = hc_eVnm/energy_eV ;   // wavelength in nm

        LOG(LEVEL) 
            << " i " << std::setw(4) << i 
            << " energy_eV " << std::setw(10) << std::fixed << std::setprecision(3) << energy_eV  
            << " wavelength_nm " << std::setw(10) << std::fixed << std::setprecision(3) << wavelength_nm
            ;

        a[reverse ? n-1-i : i] = wavelength_nm ; 
    }
    return a ; 
}


template <typename T>
std::string X4PhysicsVector<T>::Scan(const G4PhysicsVector* vec )
{
    std::stringstream ss ; 
    
    size_t num_val = vec->GetVectorLength() ; 
    ss << " num_val " << num_val << std::endl ; 
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


template class X4PhysicsVector<float>;
template class X4PhysicsVector<double>;

