#pragma once

#include "X4_API_EXPORT.hh"
#include <string>

class G4PhysicsVector ; 
template <typename T> class GProperty ; 

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
    public:
        static std::string    Digest(const G4PhysicsVector* vec ) ; 
        static GProperty<T>* Convert(const G4PhysicsVector* vec ) ; 
    public:
        X4PhysicsVector( const G4PhysicsVector* vec );

        size_t getVectorLength() const ;
        T* getValues(bool reverse) const ;
        T* getEnergies(bool reverse) const ;
        T* getWavelengths(bool reverse) const ;

        GProperty<T>* getProperty() const ;

    private:
        void init(); 
    private:
        const G4PhysicsVector* m_vec ; 
        GProperty<T>*          m_prop ; 

};

