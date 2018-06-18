#pragma once

class GMaterialLib ; 
class Opticks ; 

#include "G4MaterialTable.hh"   // typedef std::vector<G4Material*>
#include "X4_API_EXPORT.hh"

/*
X4MaterialTable
===========================

void X4MaterialTable::Convert(GMaterialLib* mlib) 
---------------------------------------------------

Converts all G4Material from the static G4MaterialTable 
into Opticks GMaterial and adds them to the GMaterialLib
in original creation order with original indices.

* mlib is expected to be empty when this is called.

*/

class X4_API X4MaterialTable 
{
    public:
        static void Convert(GMaterialLib* mlib) ; 
    private:
        X4MaterialTable(GMaterialLib* mlib);
        GMaterialLib* getMaterialLib();
        void init();
    private:
        const G4MaterialTable*  m_mtab ; 
        GMaterialLib*           m_mlib ;         
};

