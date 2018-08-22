#pragma once

#include "X4_API_EXPORT.hh"

class G4MaterialPropertiesTable ; 
template <typename T> class GPropertyMap ; 

/**
X4MaterialPropertiesTable 
===========================

Converts properties from G4MaterialPropertiesTable into the
GPropertyMap<float> base of GMaterial, GSkinSurface or GBorderSurface.

The _OLD methods give runtime warnings with 10.4.2 threatening that 
map accessors will be removed in 11

**/

class X4_API X4MaterialPropertiesTable 
{
    public:
        static void Convert(GPropertyMap<float>* pmap,  const G4MaterialPropertiesTable* const mpt);
        static std::string Digest(const G4MaterialPropertiesTable* mpt);
        static std::string Digest_OLD(const G4MaterialPropertiesTable* mpt);
    private:
        X4MaterialPropertiesTable(GPropertyMap<float>* pmap,  const G4MaterialPropertiesTable* const mpt);
        void init();
    private:
        static void AddProperties(GPropertyMap<float>* pmap, const G4MaterialPropertiesTable* const mpt);
        static void AddProperties_OLD(GPropertyMap<float>* pmap, const G4MaterialPropertiesTable* const mpt);
    private:
        GPropertyMap<float>*                   m_pmap ; 
        const G4MaterialPropertiesTable* const m_mpt ;

};
