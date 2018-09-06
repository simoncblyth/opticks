#pragma once

#include "X4_API_EXPORT.hh"

class G4MaterialPropertiesTable ; 
template <typename T> class GPropertyMap ; 
class GMaterialLib ; 

/**
X4PropertyMap
================

Convert Opticks GGeo property map into Geant4 material properties table


**/

class X4_API X4PropertyMap
{
    public:
        static G4MaterialPropertiesTable* Convert( const GPropertyMap<float>* pmap );
    public:
        X4PropertyMap(const GPropertyMap<float>* pmap) ; 
        G4MaterialPropertiesTable* getMPT() const ;
    private:
        void init(); 
    private:
        const GPropertyMap<float>*   m_pmap ; 
        G4MaterialPropertiesTable*   m_mpt ;   
        const GMaterialLib*          m_mlib ; 
};


