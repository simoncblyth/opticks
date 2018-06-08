#pragma once

class GMaterialLib ; 
class Opticks ; 

#include "G4MaterialTable.hh"   // typedef std::vector<G4Material*>
#include "X4_API_EXPORT.hh"

/*
X4MaterialTable
===========================

G4MaterialTable is a typedef to a vector of G4Material*, 
the Opticks analog is GMaterialLib.  X4MaterialTable 
converts to Opticks.


*/

class X4_API X4MaterialTable 
{
    public:
        static GMaterialLib* Convert(GMaterialLib* mlib=NULL) ; 
        static GMaterialLib* Convert(const G4MaterialTable* mtab, GMaterialLib* mlib=NULL) ; 
    public:
        X4MaterialTable(const G4MaterialTable* mtab, GMaterialLib* mlib=NULL);
        GMaterialLib*  getMaterialLib();
    private:
        void init();
    private:
        const G4MaterialTable*  m_mtab ; 
        Opticks*                m_ok ; 
        GMaterialLib*           m_mlib ;         
};

