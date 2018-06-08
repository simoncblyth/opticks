#pragma once

#include "X4_API_EXPORT.hh"

class G4Material ; 
class G4MaterialPropertiesTable ; 
class GMaterial ; 

/**
X4Material
===========

**/

class X4_API X4Material
{
    public:
        static GMaterial* Convert(const G4Material* material);
    public:
        X4Material(const G4Material* material); 
        GMaterial* getMaterial();
    private:
        void init();
    private:
        const G4Material*                m_material ;  
        const G4MaterialPropertiesTable* m_mpt ; 
        GMaterial*                       m_mat ; 
};

