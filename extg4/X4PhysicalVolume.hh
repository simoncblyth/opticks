#pragma once

#include "X4_API_EXPORT.hh"

class G4LogicalVolume ; 
class G4VPhysicalVolume ; 
#include "G4Transform3D.hh"

class GGeo ; 
class GMaterialLib ; 
class Opticks ; 

/**
X4PhysicalVolume
===========

**/

class X4_API X4PhysicalVolume
{
    public:
        static GGeo* Convert(const G4VPhysicalVolume* top);
    public:
        X4PhysicalVolume(const G4VPhysicalVolume* pv); 
        GGeo* getGGeo();
    private:
        void init();
    private:
        void VolumeTreeTraverse(); 
        G4Transform3D VolumeTreeTraverse(const G4LogicalVolume* const volumePtr, const G4int depth);
        void Visit(const G4LogicalVolume* const lv);
        void VisitPV(const G4VPhysicalVolume* const pv, const G4Transform3D& T );
    private:
        const G4VPhysicalVolume*     m_top ;  
        Opticks*                     m_ok ; 
        GGeo*                        m_ggeo ; 
        GMaterialLib*                m_mlib ; 
        int                          m_verbosity ; 
        int                          m_pvcount ; 
};

