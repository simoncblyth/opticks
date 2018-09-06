#pragma once

#include "X4_API_EXPORT.hh"

class G4MaterialPropertiesTable ; 
template <typename T> class GPropertyMap ; 

/**
X4MaterialLib
================

NB not a full conversion, just replaces Geant4 material MPT with 
the standardized domain properties from the Opticks GMaterialLib

**/

class X4_API X4MaterialLib
{
    public:
        X4MaterialLib(const GMaterialLib* mlib) ; 
    private:
        const GMaterialLib*   m_mlib ; 
};


