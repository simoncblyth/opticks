#pragma once

#include <vector>
#include <string>
#include <map>

class G4Material ; 
#include "G4MaterialTable.hh"
#include "CFG4_API_EXPORT.hh"

/**
CMaterialSort
==============

Sort the G4MaterialTable into the order specified
by the ctor argument.

BUT that makes the position of the G4Material
inconsistent with the fIndexInTable set in ctor and 
used from dtor::

    253   // Remove this material from theMaterialTable.
    254   //
    255   theMaterialTable[fIndexInTable] = 0;
    256 }

This means will be nullification of the wrong object 
on deleting G4Material.  But as there is not much 
reason to ever delete a material... lets see if it causes
any issue.

**/

class CFG4_API CMaterialSort {
        typedef std::map<std::string, unsigned> MSU ; 
   public:
        CMaterialSort(const std::map<std::string, unsigned>& order ); 
        bool operator()(const G4Material* a, const G4Material* b) ;
   private:
        void init();      
        void dump(const char* msg) const ;      
        void dumpOrder(const char* msg) const ;      
        void sort();      
   private:
        const std::map<std::string, unsigned>&  m_order  ;
        G4MaterialTable*                        m_mtab ; 
};

 
