#pragma once

#include <map>

class G4LogicalSurface ; 
class G4LogicalVolume ; 
class G4VPhysicalVolume ; 
class G4VSolid ; 
#include "G4Transform3D.hh"

#include "NGLM.hpp"
#include "X4_API_EXPORT.hh"

struct nxform ; 

class GGeo ; 
class GMaterialLib ; 
class GSurfaceLib ; 
class GBndLib ; 
class GVolume ; 
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

Hmm this shoud probably be named X4Scene or X4Tree, 
as it forcusses on the tree not the PhysicalVolume node.

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
        GVolume* convertTree_r(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, int preorder, const G4VPhysicalVolume* const parent_pv );
        GVolume* convertNode(const G4VPhysicalVolume* const pv, int depth, int preorder, const G4VPhysicalVolume* const parent_pv );
        void convertSolid( YOG::Mh* mh,  const G4VSolid* const solid);
        G4LogicalSurface* findSurface( const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_priority );
    private:
        const G4VPhysicalVolume*     m_top ;  
        const char*                  m_key ;  
        bool                         m_keyset ; 
        Opticks*                     m_ok ; 
        const char*                  m_gltfpath ; 
    private:
        GGeo*                        m_ggeo ; 
        GMaterialLib*                m_mlib ; 
        GSurfaceLib*                 m_slib ; 
        GBndLib*                     m_blib ; 
    private:
        GVolume*                     m_root ;  
    private:
        nxform*                      m_xform ; 
    private:
        YOG::Sc*                     m_sc ; 
        YOG::Maker*                  m_maker ; 
        int                          m_verbosity ; 

        std::map<const G4LogicalVolume* const, int> m_lvidx ; 


};

