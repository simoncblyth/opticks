#pragma once

class Opticks ; 
class G4VPhysicalVolume ;
class G4LogicalVolume ;
class G4LogicalBorderSurface ;

/**
CCheck
========

Recursively traverses a Geant4 geometry tree, 
checking integrity. Loosely follows access pattern
of G4GDMLWriter.

**/


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CCheck {
    public:
        CCheck(Opticks* ok, G4VPhysicalVolume* top);
    public:
        void checkSurf();
    private:
        void checkSurfTraverse(const G4LogicalVolume* const lv, const int depth);
    private:
        const G4LogicalBorderSurface* GetBorderSurface(const G4VPhysicalVolume* const pvol) ;
    private:
        Opticks*                       m_ok ; 
        G4VPhysicalVolume*             m_top ; 
};

#include "CFG4_TAIL.hh"


