#pragma once

#include <map>

class G4LogicalSurface ; 
class G4LogicalVolume ; 
class G4VPhysicalVolume ; 
class G4VSolid ; 
#include "G4Transform3D.hh"

#include "NGLM.hpp"
#include "X4_API_EXPORT.hh"

class GGeo ; 
class GMaterialLib ; 
class GSurfaceLib ; 
class GBndLib ; 
class Opticks ; 

namespace YOG 
{
   struct Sc ; 
   struct Mh ; 
   struct Nd ; 
   struct Maker ; 
}

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
        void  saveAsGLTF(const char* path);
    private:
        void init();
    private:
        void convertMaterials(); 
        void convertSurfaces(); 
        void convertStructure(); 
    private:
        void IndexTraverse(const G4VPhysicalVolume* const pv, int depth);
    private:
        int TraverseVolumeTree(const G4VPhysicalVolume* const pv, int depth, int preorder, const G4VPhysicalVolume* const parent_pv );
        YOG::Nd* convertNodeVisit(const G4VPhysicalVolume* const pv, int depth, const G4VPhysicalVolume* const parent_pv );
        int  convertMaterialVisit(const G4Material* const material );
        void convertSolid( YOG::Mh* mh,  G4VSolid* solid);
        G4LogicalSurface* findSurface( const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_priority );
    private:
        const G4VPhysicalVolume*     m_top ;  
        const char*                  m_key ;  
        bool                         m_keyset ; 
        Opticks*                     m_ok ; 
    private:
        GGeo*                        m_ggeo ; 
        GMaterialLib*                m_mlib ; 
        GSurfaceLib*                 m_slib ; 
        GBndLib*                     m_blib ; 
    private:
        YOG::Sc*                     m_sc ; 
        YOG::Maker*                  m_maker ; 
        int                          m_verbosity ; 
        int                          m_pvcount ; 
        glm::mat4                    m_identity ; 

        std::map<const G4LogicalVolume* const, int> m_lvidx ; 


};

