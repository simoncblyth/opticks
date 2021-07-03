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

#pragma once

#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"
#include <string>

class G4PhysicsVector ; 
template <typename T> class GProperty ; 
template <typename T> class GDomain ; 

/**
X4PhysicsVector
===================

Converts G4PhysicsVector into ggeo.GProperty<T>

1. input G4PhysicsVector assumed to use standard Geant4 energy units, 
   with energies in ascending order.

2. output ggeo.GProperty<T> uses wavelength domain in nanometers 
   in wavelength ascending order (which is reversed wrt energy)

**/

template <typename T>
class X4_API X4PhysicsVector
{
        static const plog::Severity LEVEL ; 
    public:
        static std::string    Digest0(const G4PhysicsVector* vec ) ; 
        static std::string    Digest(const G4PhysicsVector* vec ) ; 
        static std::string    Scan(const G4PhysicsVector* vec ) ; 
    public:
        static T _hc_eVnm() ; 
        static GProperty<T>* Convert(const G4PhysicsVector* vec, bool nm_domain ) ; 
        static GProperty<T>* Interpolate(const G4PhysicsVector* vec,  const GDomain<T>* dom ) ; 
    private:
        X4PhysicsVector( const G4PhysicsVector* vec, const GDomain<T>* dom, bool nm_domain  );
        void init(); 
    private:
        GProperty<T>* makeDirect(bool nm_domain) const ; 
        size_t getSrcVectorLength() const ;
        T* getSrcValues(bool reverse) const ;
        T* getSrcEnergies(bool reverse) const ;
        T* getSrcWavelengths(bool reverse) const ;
    private:
        GProperty<T>* makeInterpolated() const ; 
        T* getInterpolatedValues(T* wavelength_nm, size_t n, T hc_eVnm=0. ) const ;   // set hc_eVnm to 1240. to use that instead of more precise one
    private:
        GProperty<T>* getProperty() const ;
    private:
        const G4PhysicsVector* m_vec ; 
        const GDomain<T>*      m_dom ; 
        GProperty<T>*          m_prop ; 

};

