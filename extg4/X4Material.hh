#pragma once

#include "X4_API_EXPORT.hh"
#include <vector>
#include <string>

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
        static std::string Digest();
        static std::string Digest(const std::vector<G4Material*>& materials);
        static std::string Digest(const G4Material* material);
        static std::string Digest(const G4MaterialPropertiesTable* mpt);
    public:
        static GMaterial* Convert(const G4Material* material);
        static void       AddProperties(GMaterial* mat, const G4MaterialPropertiesTable* mpt);
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

