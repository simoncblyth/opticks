#pragma once

class G4MaterialPropertiesTable ; 

#include "X4_API_EXPORT.hh"

/*
X4MaterialPropertiesTable
===========================

*/

template <typename T> class GPropertyMap ;


class X4_API X4MaterialPropertiesTable 
{
    public:
        static GPropertyMap<float>* Convert(const G4MaterialPropertiesTable*) ; 
    public:
        X4MaterialPropertiesTable(const G4MaterialPropertiesTable* mpt);
        GPropertyMap<float>*  getPropertyMap();
    private:
        void init();
    private:
        const G4MaterialPropertiesTable* m_mpt ; 
        GPropertyMap<float>*             m_pmap ;         

};

