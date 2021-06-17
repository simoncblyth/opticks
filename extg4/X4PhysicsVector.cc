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
    GProperty<T>* prop = Convert( vec ) ; 
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

    LOG(LEVEL) << " len " << len ; 

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

template <typename T>
T* X4PhysicsVector<T>::getEnergies(bool reverse) const
{
    size_t n = getVectorLength() ; 
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

        LOG(LEVEL) 
            << " i " << std::setw(4) << i 
            << " energy " << std::setw(10) << energy   
            << " wavelength " << std::setw(10) << wavelength   
            ;

        a[reverse ? n-1-i : i] = wavelength ; 
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

