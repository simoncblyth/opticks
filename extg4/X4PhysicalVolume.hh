/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include "plog/Severity.h"

class G4LogicalSurface ; 
class G4LogicalVolume ; 
class G4VPhysicalVolume ; 
class G4VSolid ; 
class G4Material ; 
class G4PVPlacement ; 

#include "G4Transform3D.hh"

struct nnode ; 
#include "NGLM.hpp"
#include "X4_API_EXPORT.hh"
#include "X4Named.hh"


//#define X4_TRANSFORM 1


template <typename T> struct nxform ; 

class GGeo ; 
class GMesh ; 
class GMaterialLib ; 
class GSurfaceLib ; 
class GBndLib ; 
class GMeshLib ; 
class GVolume ; 

class Opticks ; 
class OpticksQuery ; 

/**
X4PhysicalVolume
===================

Constructor populates the GGeo instance via direct conversion of 
materials, surfaces and structure from the passed world volume::

    X4PhysicalVolume(GGeo* ggeo, const G4VPhysicalVolume* const pv); 

TODO: rename to something like X4Scene/X4Tree/X4Top/X4Root
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
class SDigest ; 

struct X4_API X4Nd 
{
    const X4Nd*         parent ; 
    const nmat4triple*  transform ; 
};


class X4_API X4PhysicalVolume : public X4Named 
{
    public:
        static const char* TMPDIR ; 
        static const plog::Severity  LEVEL ; 
        static const bool            DBG ; 
        static const G4VPhysicalVolume* const Top();
        static GGeo* Convert(const G4VPhysicalVolume* const top);
    public:
        // digestextra is a kludge for things like csgskiplv arguments that change geometry, 
        // see notes/issues/opticks_key_digest_no_updating_for_changed_geometry.rst
        static const char* Key(const G4VPhysicalVolume* const top, const char* digestextra=NULL, const char* digestextra2=NULL);
        static std::string Digest( const G4VPhysicalVolume* const top, const char* digestextra=NULL, const char* digestextra2=NULL);
        static void        Digest( const G4LogicalVolume* const lv, const G4int depth, SDigest* dig );
        std::string brief() const ;
    public:
        static void DumpSensorVolumes(const GGeo* ggeo, const char* msg); 
        static void GetSensorPlacements(const GGeo* gg, std::vector<G4PVPlacement*>& placements);
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
        void convertStructureChecks() const ;
    private:
        void convertCheck() const ;
        void postConvert() const ; 
    private:
        bool hasEfficiency(const G4Material* mat);
    private:
        void convertSolids_r(const G4VPhysicalVolume* const pv, int depth);
        void dumpLV() const ;
        void dumpTorusLV() const ;
        GMesh* convertSolid( int lvIdx, int soIdx, const G4VSolid* const solid, const std::string& lvname, bool balancetree ) const ;
        void generateTestG4Code( int lvIdx, const G4VSolid* const solid, const nnode* raw) const ; 
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
        GMeshLib*                         m_hlib ; 
 
        //const std::vector<const GMesh*>&  m_meshes ;  
    private:
        GVolume*                     m_root ;  
    private:
        nxform<X4Nd>*                m_xform ; 
    private:
        int                          m_verbosity ; 
        unsigned                     m_node_count ; 
        unsigned                     m_selected_node_count ; 
#ifdef X4_PROFILE    
    private:
        float                        m_convertNode_dt ;  
        float                        m_convertNode_boundary_dt ;  
        float                        m_convertNode_transformsA_dt ;  
        float                        m_convertNode_transformsB_dt ;  
        float                        m_convertNode_transformsC_dt ;  
        float                        m_convertNode_transformsD_dt ;  
        float                        m_convertNode_transformsE_dt ;  
        float                        m_convertNode_GVolume_dt ;  
#endif  
#ifdef X4_TRANSFORM
        int                          m_is_identity0 ; 
        int                          m_is_identity1 ; 
#endif
        int                          m_dummy ;   
    private:
        std::map<const G4LogicalVolume*, int> m_lvidx ; 
        std::vector<const G4LogicalVolume*>   m_lvlist ; 
        std::vector<unsigned>                 m_lv_with_torus ; 
        std::vector<std::string>              m_lvname ; 
        std::vector<std::string>              m_soname ; 
        std::vector<std::string>              m_lvname_with_torus ; 
        std::vector<std::string>              m_soname_with_torus ; 
        std::vector<G4Material*>              m_material_with_efficiency ; 

};

