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
===================



CAUTION regarding geometry digests
------------------------------------

The Digest methods provide strings that are used to represent the identity 
of the geometry.  However they are a long way from being complete 
digests : ie many types of geometry changes will not result in a different
digest.  The identity string is however just used to provide 
a name for the geometry cache.  


**/

class X4_API X4PhysicalVolume
{
    public:
        static const G4VPhysicalVolume* const Top();
        static GGeo* Convert(const G4VPhysicalVolume* const top);
    public:
        static const char* Key(const G4VPhysicalVolume* const top);
        static std::string Digest( const G4VPhysicalVolume* const top);
        static std::string Digest( const G4LogicalVolume* const lv, const G4int depth );
    public:
        X4PhysicalVolume(const G4VPhysicalVolume* const pv); 
        GGeo* getGGeo();
    private:
        void init();
    private:
        void TraverseVolumeTree(); 
        G4Transform3D TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth);
        void Visit(const G4LogicalVolume* const lv);
        void VisitPV(const G4VPhysicalVolume* const pv, const G4Transform3D& T );
    private:
        const G4VPhysicalVolume*     m_top ;  
        const char*                  m_key ;  
        bool                         m_keyset ; 
        Opticks*                     m_ok ; 
        GGeo*                        m_ggeo ; 
        GMaterialLib*                m_mlib ; 
        int                          m_verbosity ; 
        int                          m_pvcount ; 
};

