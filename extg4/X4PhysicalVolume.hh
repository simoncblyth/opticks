#pragma once

#include <string>
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
class GMesh ; 
class GMaterialLib ; 
class GSurfaceLib ; 
class GBndLib ; 
class GVolume ; 

class Opticks ; 
class OpticksQuery ; 

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

struct nmat4triple ; 

struct X4_API X4Nd 
{
    const X4Nd*         parent ; 
    const nmat4triple*  transform ; 
};


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
    private:
        void init();
    private:
        void convertMaterials(); 
        void convertSurfaces(); 
        void convertSensors(); 
        void closeSurfaces(); 
        void convertSolids(); 
        void convertStructure(); 
    private:
        void convertSolids_r(const G4VPhysicalVolume* const pv, int depth);
        void dumpLV();
        GMesh* convertSolid( int lvIdx, int soIdx, const G4VSolid* const solid, const std::string& lvname) const ;
    private:
        void convertSensors_r(const G4VPhysicalVolume* const pv, int depth);
        GVolume* convertStructure_r(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const parent_pv, bool& recursive_select );
        GVolume* convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const parent_pv, bool& recursive_select );
        unsigned addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p );

        G4LogicalSurface* findSurface( const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_priority );
    private:
        GGeo*                        m_ggeo ; 
        const G4VPhysicalVolume*     m_top ;  
        Opticks*                     m_ok ; 
        const char*                  m_lvsdname ; 
        OpticksQuery*                m_query ; 
        const char*                  m_gltfpath ; 
        bool                         m_g4codegen ; 
        const char*                  m_g4codegendir ;
    private:
        GMaterialLib*                m_mlib ; 
        GSurfaceLib*                 m_slib ; 
        GBndLib*                     m_blib ; 
    private:
        GVolume*                     m_root ;  
    private:
        nxform<X4Nd>*                m_xform ; 
    private:
        int                          m_verbosity ; 
        unsigned                     m_node_count ; 

        std::map<const G4LogicalVolume*, int> m_lvidx ; 

        std::vector<const G4LogicalVolume*> m_lvlist ; 




};

