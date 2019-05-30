#pragma once

#include "X4_API_EXPORT.hh"

template <typename T> class GPropertyMap ; 
class GMaterialLib ;
#include "G4MaterialTable.hh"

/**
X4MaterialLib
================

NB not a full conversion, just replaces Geant4 material MPT with 
the standardized domain properties from the Opticks GMaterialLib

**/

class X4_API X4MaterialLib
{
    public:
        static void Standardize(); 
        static void Standardize( G4MaterialTable* mtab, const GMaterialLib* mlib ); 
    public:
        X4MaterialLib(G4MaterialTable* mtab,  const GMaterialLib* mlib) ; 
    private:
        void init(); 
    private:
        G4MaterialTable*      m_mtab ; 
        const GMaterialLib*   m_mlib ; 
};


