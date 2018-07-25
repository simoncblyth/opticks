#pragma once

#include <vector>
#include <map>

class G4LogicalSurface ; 
class G4LogicalVolume ; 
class G4VPhysicalVolume ; 
class G4VSolid ; 
#include "G4Transform3D.hh"

#include "NGLM.hpp"
#include "X4_API_EXPORT.hh"


template <typename T> struct nxform ; 

class GGeo ; 
class GMaterialLib ; 
class GSurfaceLib ; 
class GBndLib ; 
class GVolume ; 

class Opticks ; 
class OpticksQuery ; 

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

Constructor populates the GGeo instance via direct conversion of 
materials, surfaces and structure from the passed world volume::

    X4PhysicalVolume(GGeo* ggeo, const G4VPhysicalVolume* const pv); 


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
        X4PhysicalVolume(GGeo* ggeo, const G4VPhysicalVolume* const pv); 
        GGeo* getGGeo();
        void  saveAsGLTF(int root=0, const char* path=NULL);
    private:
        void init();
    private:
        void convertMaterials(); 
        void convertSurfaces(); 
        void convertStructure(); 
    private:
        void IndexTraverse(const G4VPhysicalVolume* const pv, int depth);
        void dumpLV();
    private:
        GVolume* convertTree_r(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const parent_pv, bool& recursive_select );
        GVolume* convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const parent_pv, bool& recursive_select );
        unsigned addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p );
        void convertSolid( YOG::Mh* mh,  const G4VSolid* const solid);
        G4LogicalSurface* findSurface( const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_priority );
    private:
        GGeo*                        m_ggeo ; 
        const G4VPhysicalVolume*     m_top ;  
        Opticks*                     m_ok ; 
        OpticksQuery*                m_query ; 
        const char*                  m_gltfpath ; 
    private:
        GMaterialLib*                m_mlib ; 
        GSurfaceLib*                 m_slib ; 
        GBndLib*                     m_blib ; 
    private:
        GVolume*                     m_root ;  
    private:
        nxform<YOG::Nd>*             m_xform ; 
    private:
        YOG::Sc*                     m_sc ; 
        YOG::Maker*                  m_maker ; 
        int                          m_verbosity ; 
        unsigned                     m_ndCount ; 

        std::map<const G4LogicalVolume*, int> m_lvidx ; 

        std::vector<const G4LogicalVolume*> m_lvlist ; 

        



};

