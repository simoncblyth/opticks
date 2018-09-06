#pragma once

#include "X4_API_EXPORT.hh"

class G4PhysicsVector ; 
template <typename T> class GProperty ; 

/**
X4Property
===================

Converts ggeo.GProperty<T> into G4PhysicsVector

1. input ggeo.GProperty<T> uses wavelength domain in nanometers 
   in wavelength ascending order (which is reversed wrt energy)

2. output G4PhysicsVector using standard Geant4 energy units (MeV)
   with energies in ascending order.

**/

template <typename T>
class X4_API X4Property
{
    public:
        static G4PhysicsVector*  Convert(const GProperty<T>* prop) ; 
    public:
        X4Property( const GProperty<T>* prop );
        G4PhysicsVector* getVector() const ;
    private:
        void init(); 
    private:
        const GProperty<T>*    m_prop ; 
        G4PhysicsVector*       m_vec ; 

};

