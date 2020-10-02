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

#include <iostream>
#include <sstream>
#include <iomanip>

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4Material.hh"
//#include "G4VisExtent.hh"
#include "G4VSolid.hh"
#include "G4TransportationManager.hh"
#include "G4VSensitiveDetector.hh"
#include "G4PVPlacement.hh"

#include "X4.hh"
#include "X4PhysicalVolume.hh"
#include "X4Material.hh"
#include "X4MaterialTable.hh"
#include "X4LogicalBorderSurfaceTable.hh"
#include "X4LogicalSkinSurfaceTable.hh"
#include "X4Solid.hh"
#include "X4CSG.hh"
#include "X4GDMLParser.hh"
#include "X4Mesh.hh"
#include "X4Transform3D.hh"

#include "SStr.hh"
#include "SSys.hh"
#include "SDigest.hh"
#include "SGDML.hh"
#include "PLOG.hh"

#include "BStr.hh"
#include "BFile.hh"
#include "BTimeStamp.hh"
#include "BOpticksKey.hh"

class NSensor ; 

#include "NXform.hpp"  // header with the implementation
template struct nxform<X4Nd> ; 

#include "NGLMExt.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NNodeNudger.hpp"
#include "NTreeProcess.hpp"
#include "GLMFormat.hpp"

#include "GMesh.hh"
#include "GVolume.hh"

#ifdef GPARTS_HOT
#include "GParts.hh"
#endif

#include "GPt.hh"
#include "GPts.hh"

#include "GGeo.hh"
#ifdef OLD_SENSOR
#include "GGeoSensor.hh"
#endif
#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GSkinSurface.hh"
#include "GBndLib.hh"
#include "GMeshLib.hh"

#include "Opticks.hh"
#include "OpticksQuery.hh"

/**
X4PhysicalVolume
==================


**/


const plog::Severity X4PhysicalVolume::LEVEL = PLOG::EnvLevel("X4PhysicalVolume", "DEBUG") ;
const bool           X4PhysicalVolume::DBG = true ;



const G4VPhysicalVolume* const X4PhysicalVolume::Top()
{
    const G4VPhysicalVolume* const top = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;
    return top ; 
}

GGeo* X4PhysicalVolume::Convert(const G4VPhysicalVolume* const top)
{
    const char* key = X4PhysicalVolume::Key(top) ; 

    BOpticksKey::SetKey(key);

    LOG(error) << " SetKey " << key  ; 

    Opticks* ok = new Opticks(0,0);  // Opticks instanciation must be after BOpticksKey::SetKey

    ok->configure();   // see notes/issues/configuration_resource_rejig.rst

    GGeo* gg = new GGeo(ok) ;

    X4PhysicalVolume xtop(gg, top) ;   // <-- populates gg 

    return gg ; 
}




X4PhysicalVolume::X4PhysicalVolume(GGeo* ggeo, const G4VPhysicalVolume* const top)
    :
    X4Named("X4PhysicalVolume"),
    m_ggeo(ggeo),
    m_top(top),
    m_ok(m_ggeo->getOpticks()), 
    m_lvsdname(m_ok->getLVSDName()),
    m_query(m_ok->getQuery()),
    m_gltfpath(m_ok->getGLTFPath()),
    m_g4codegen(m_ok->isG4CodeGen()),
    m_g4codegendir(m_ok->getG4CodeGenDir()),
    m_mlib(m_ggeo->getMaterialLib()),
    m_slib(m_ggeo->getSurfaceLib()),
    m_blib(m_ggeo->getBndLib()),
    m_hlib(m_ggeo->getMeshLib()),
    //m_meshes(m_hlib->getMeshes()), 
    m_xform(new nxform<X4Nd>(0,false)),
    m_verbosity(m_ok->getVerbosity()),
    m_node_count(0),
    m_selected_node_count(0),
#ifdef X4_PROFILE
    m_convertNode_dt(0.f),
    m_convertNode_boundary_dt(0.f),
    m_convertNode_transformsA_dt(0.f),
    m_convertNode_transformsB_dt(0.f),
    m_convertNode_transformsC_dt(0.f),
    m_convertNode_transformsD_dt(0.f),
    m_convertNode_transformsE_dt(0.f),
    m_convertNode_GVolume_dt(0.f),
#endif
#ifdef X4_TRANSFORM
    m_is_identity0(0),
    m_is_identity1(0),
#endif
    m_dummy(0)
{
    const char* msg = "GGeo ctor argument of X4PhysicalVolume must have mlib, slib, blib and hlib already " ; 

    // trying to Opticks::configure earlier, from Opticks::init trips these asserts
    assert( m_mlib && msg ); 
    assert( m_slib && msg ); 
    assert( m_blib && msg ); 
    assert( m_hlib && msg ); 

    init();
}

GGeo* X4PhysicalVolume::getGGeo()
{
    return m_ggeo ; 
}

void X4PhysicalVolume::init()
{
    LOG(LEVEL) << "[" ; 
    LOG(LEVEL) << " query : " << m_query->desc() ; 


    convertMaterials();   // populate GMaterialLib
    convertSurfaces();    // populate GSurfaceLib
#ifdef OLD_SENSOR
    convertSensors();  // before closeSurfaces as may add some SensorSurfaces
#endif
    closeSurfaces();
    convertSolids();      // populate GMeshLib with GMesh converted from each G4VSolid (postorder traverse processing first occurrence of G4LogicalVolume)  
    convertStructure();   // populate GNodeLib with GVolume converted from each G4VPhysicalVolume (preorder traverse) 
    convertCheck();

    postConvert(); 

    LOG(LEVEL) << "]" ; 
}



void X4PhysicalVolume::postConvert() const 
{
    LOG(info) 
        << " GGeo::getNumVolumes() " << m_ggeo->getNumVolumes() 
        << " GGeo::getNumSensorVolumes() " << m_ggeo->getNumSensorVolumes() 
        << std::endl 
        << " GGeo::getSensorBoundaryReport() "
        << std::endl 
        << m_ggeo->getSensorBoundaryReport()
        ;

    // too soon for sensor dumping as instances not yet formed, see GGeo::postDirectTranslationDump 
    //m_ggeo->dumpSensorVolumes("X4PhysicalVolume::postConvert"); 

}



#ifdef OLD_SENSOR

/**
X4PhysicalVolume::convertSensors
---------------------------------

Predecessor in old route is AssimpGGeo::convertSensors

* note the recursive call X4PhysicalVolume::convertSensors_r 
  which traverses the geometry looking for sensors  
  
**/

void X4PhysicalVolume::convertSensors()
{
    assert(0) ; // MATERIAL-CENTRIC APPROACH NO LONGER USED : NOW DETECTING SENSORS VIA SURFACES WITH EFFICIENCY 
    LOG(debug) << "[" ; 

    convertSensors_r(m_top, 0); 

    unsigned num_clv = m_ggeo->getNumCathodeLV();
    unsigned num_bds = m_ggeo->getNumBorderSurfaces() ; 
    unsigned num_sks0 = m_ggeo->getNumSkinSurfaces() ; 

    GGeoSensor::AddSensorSurfaces(m_ggeo) ;

    unsigned num_sks1 = m_ggeo->getNumSkinSurfaces() ; 
    assert( num_bds == m_ggeo->getNumBorderSurfaces()  ); 

    unsigned num_lvsd = m_ggeo->getNumLVSD() ; 

    LOG(debug) << "]" ; 

    LOG(info) 
         << " m_lvsdname " << m_lvsdname 
         << " num_lvsd " << num_lvsd 
         << " num_clv " << num_clv 
         << " num_bds " << num_bds
         << " num_sks0 " << num_sks0
         << " num_sks1 " << num_sks1
         ; 


}

/**
X4PhysicalVolume::convertSensors_r
-----------------------------------

Recurses over the geometry looking for volumes with associated SensitiveDetector, 
when found invokes GGeo::addLVSD persisting the association between an LV name 
and an SD name.

Sensors are identified by two approaches:

1. logical volume having an associated sensitive detector G4VSensitiveDetector
2. name of logical volume matching one of a comma delimited list 
   of strings provided by the "LV sensitive detector name" option
   eg  "--lvsdname Cathode,cathode,Sensor,SD" 

The second approach is useful as a workaround when operating 
with a GDML loaded geometry, as GDML does not yet(?) persist 
the SD LV association.

Names of sensitive LV are inserted into a set datastructure in GGeo. 



Issues/TODO
~~~~~~~~~~~~~

* how to flexibly associate an EFFICIENCY property
  with the sensor ?

  * currently a simplifying assumption of a single "Cathode" 
    material is made : how to generalize ?

**Possible Generalization Approach**

First thing to try is extend GGeo::addLVSD to GGeo::addLVSDMT
which collects material names which are asserted to hold a
an EFFICIENCY property. These materials can then replace the 
GMaterialLib::setCathode getCathode

* If the efficiency is zero should a sensor be created ?
 

**/

void X4PhysicalVolume::convertSensors_r(const G4VPhysicalVolume* const pv, int depth)
{
    assert(0) ; // MATERIAL-CENTRIC APPROACH NO LONGER USED : NOW DETECTING SENSORS VIA SURFACES WITH EFFICIENCY 


    // hot code : minimize whats done outside the if

    const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ; 

    const G4Material* const mt = lv->GetMaterial() ;
    bool has_efficiency = hasEfficiency(mt) ;  

    const char* lvname = lv->GetName().c_str(); 
    bool is_lvsdname = m_lvsdname && BStr::Contains(lvname, m_lvsdname, ',' ) ;
    bool is_sd = sd != NULL || is_lvsdname ; 


    if( is_sd || has_efficiency )
    {
        const std::string sdn = sd ? sd->GetName() : "SD?" ;   // perhaps GetFullPathName() 
        const char* mt_name = mt->GetName().c_str(); 

        std::string name = BFile::Name(lvname); 
        bool addPointerToName = false ;   // <-- maybe this should depend on if booting from GDML or not ?
        std::string nameref = SGDML::GenerateName( name.c_str() , lv , addPointerToName );   

        LOG(LEVEL) 
            << " is_lvsdname " << is_lvsdname
            << " is_sd " << is_sd
            << " has_efficiency " << has_efficiency
            << " sdn " << sdn 
            << " name " << name 
            << " nameref " << nameref 
            << " mt_name " << mt_name
            ;
 
        m_ggeo->addLVSDMT(nameref.c_str(), sdn.c_str(), mt_name ) ;
    }  

    for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    {
        const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
        convertSensors_r(child_pv, depth+1 );
    }
}

#endif


bool X4PhysicalVolume::hasEfficiency(const G4Material* mat)
{
    return std::find(m_material_with_efficiency.begin(), m_material_with_efficiency.end(), mat ) != m_material_with_efficiency.end() ;
}


void X4PhysicalVolume::convertMaterials()
{
    OK_PROFILE("_X4PhysicalVolume::convertMaterials");
    LOG(verbose) << "[" ;

    size_t num_materials0 = m_mlib->getNumMaterials() ;
    assert( num_materials0 == 0 );

    assert( m_material_with_efficiency.size() == 0 );
    X4MaterialTable::Convert(m_mlib, m_material_with_efficiency);
    size_t num_material_with_efficiency = m_material_with_efficiency.size() ;


    size_t num_materials = m_mlib->getNumMaterials() ;
    assert( num_materials > 0 );

    // Adding test materials only at Opticks level is a standardization
    // problem : TODO: implement creation of test materials at G4 level
    // then they will be present at all levels.
    // 
    //m_mlib->addTestMaterials() ;

    m_mlib->close();   // may change order if prefs dictate

    LOG(verbose) << "]" ;
    LOG(info)
          << " num_materials " << num_materials
          << " num_material_with_efficiency " << num_material_with_efficiency
          ; 

    m_mlib->dumpSensitiveMaterials("X4PhysicalVolume::convertMaterials");

    OK_PROFILE("X4PhysicalVolume::convertMaterials");
}

/**
X4PhysicalVolume::convertSurfaces
-------------------------------------

* G4LogicalSkinSurface -> GSkinSurface -> slib
* G4LogicalBorderSurface -> GBorderSurface -> slib


**/

void X4PhysicalVolume::convertSurfaces()
{
    LOG(verbose) << "[" ;

    size_t num_surf0 = m_slib->getNumSurfaces() ; 
    assert( num_surf0 == 0 );

    X4LogicalBorderSurfaceTable::Convert(m_slib);
    size_t num_lbs = m_slib->getNumSurfaces() ; 

    X4LogicalSkinSurfaceTable::Convert(m_slib);
    size_t num_sks = m_slib->getNumSurfaces() - num_lbs ; 

    m_slib->addPerfectSurfaces();
    m_slib->dumpSurfaces("X4PhysicalVolume::convertSurfaces");

    m_slib->collectSensorIndices(); 
    m_slib->dumpSensorIndices("X4PhysicalVolume::convertSurfaces"); 

    LOG(verbose) << "]" ;

    LOG(info) 
           << " num_lbs " << num_lbs
           << " num_sks " << num_sks
           ;

}

void X4PhysicalVolume::closeSurfaces()
{
    m_slib->close();  // may change order if prefs dictate

#ifdef OLD_SENSOR
    m_ggeo->dumpCathodeLV("dumpCathodeLV"); 
#endif
    m_ggeo->dumpSkinSurface("dumpSkinSurface"); 
}


/**
X4PhysicalVolume::Digest
--------------------------

Looks like not succeeding to spot changes.

**/


void X4PhysicalVolume::Digest( const G4LogicalVolume* const lv, const G4int depth, SDigest* dig )
{

    for (unsigned i=0; i < unsigned(lv->GetNoDaughters()) ; i++)
    {
        const G4VPhysicalVolume* const d_pv = lv->GetDaughter(i);

        G4RotationMatrix rot, invrot;

        if (d_pv->GetFrameRotation() != 0)
        {
           rot = *(d_pv->GetFrameRotation());
           invrot = rot.inverse();
        }

        Digest(d_pv->GetLogicalVolume(),depth+1, dig);

        // postorder visit region is here after the recursive call

        G4Transform3D P(invrot,d_pv->GetObjectTranslation());

        std::string p_dig = X4Transform3D::Digest(P) ; 
    
        dig->update( const_cast<char*>(p_dig.data()), p_dig.size() );  
    }

    // Avoid pointless repetition of full material digests for every 
    // volume by digesting just the material name (could use index instead)
    // within the recursion.
    //
    // Full material digests of all properties are included after the recursion.

    G4Material* material = lv->GetMaterial();
    const G4String& name = material->GetName();    
    dig->update( const_cast<char*>(name.data()), name.size() );  

}


std::string X4PhysicalVolume::Digest( const G4VPhysicalVolume* const top, const char* digestextra, const char* digestextra2 )
{
    SDigest dig ;
    const G4LogicalVolume* lv = top->GetLogicalVolume() ;
    Digest(lv, 0, &dig ); 
    std::string mats = X4Material::Digest(); 

    dig.update( const_cast<char*>(mats.data()), mats.size() );  

    if(digestextra)
    {
        LOG(info) << "digestextra " << digestextra ; 
        dig.update_str( digestextra );  
    }
    if(digestextra2)
    {
        LOG(info) << "digestextra2 " << digestextra2 ; 
        dig.update_str( digestextra2 );  
    }


    return dig.finalize();
}


const char* X4PhysicalVolume::Key(const G4VPhysicalVolume* const top, const char* digestextra, const char* digestextra2 )
{
    std::string digest = Digest(top, digestextra, digestextra2 );
    const char* exename = PLOG::instance ? PLOG::instance->args.exename() : "OpticksEmbedded" ; 
    std::stringstream ss ; 
    ss 
       << exename
       << "."
       << "X4PhysicalVolume"
       << "."
       << top->GetName()
       << "."
       << digest 
       ;
       
    std::string key = ss.str();
    return strdup(key.c_str());
}   


/**
X4PhysicalVolume::findSurface
------------------------------

1. look for a border surface from PV_a to PV_b (do not look for the opposite direction)
2. if no border surface look for a logical skin surface with the lv of the first PV_a otherwise the lv of PV_b 
   (or vv when first_priority is false) 

**/

G4LogicalSurface* X4PhysicalVolume::findSurface( const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_priority )
{
     G4LogicalSurface* surf = G4LogicalBorderSurface::GetSurface(a, b) ;

     const G4VPhysicalVolume* const first  = first_priority ? a : b ; 
     const G4VPhysicalVolume* const second = first_priority ? b : a ; 

     if(surf == NULL)
         surf = G4LogicalSkinSurface::GetSurface(first ? first->GetLogicalVolume() : NULL );

     if(surf == NULL)
         surf = G4LogicalSkinSurface::GetSurface(second ? second->GetLogicalVolume() : NULL );

     return surf ; 
}




/**
X4PhysicalVolume::convertSolids
-----------------------------------

Uses postorder recursive traverse, ie the "visit" is in the 
tail after the recursive call, to match the traverse used 
by GDML, and hence giving the same "postorder" indices
for the solid lvIdx.

The entire volume tree is recursed, but only the 
first occurence of each LV solid gets converted 
(because they are all the same).
Done this way to have consistent lvIdx soIdx indexing with GDML ?

**/

void X4PhysicalVolume::convertSolids()
{
    OK_PROFILE("_X4PhysicalVolume::convertSolids");
    LOG(LEVEL) << "[" ; 

    const G4VPhysicalVolume* pv = m_top ; 
    int depth = 0 ;
    convertSolids_r(pv, depth);
    
    if(m_verbosity > 5) dumpLV();
    LOG(debug) << "]" ; 

    dumpTorusLV();
    dumpLV();

    LOG(LEVEL) << "]" ;
    OK_PROFILE("X4PhysicalVolume::convertSolids");

}







/**
X4PhysicalVolume::convertSolids_r
------------------------------------

G4VSolid is converted to GMesh with associated analytic NCSG 
and added to GGeo/GMeshLib.

If the conversion from G4VSolid to GMesh/NCSG/nnode required
balancing of the nnode then the conversion is repeated 
without the balancing and an alt reference is to the alternative 
GMesh/NCSG/nnode is kept in the primary GMesh. 

Note that only the nnode is different due to the balancing, however
its simpler to keep a one-to-one relationship between these three instances
for persistency convenience.

**/

void X4PhysicalVolume::convertSolids_r(const G4VPhysicalVolume* const pv, int depth)
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    {
        const G4VPhysicalVolume* const daughter_pv = lv->GetDaughter(i);
        convertSolids_r( daughter_pv , depth + 1 );
    }

    // for newly encountered lv record the tail/postorder idx for the lv
    if(std::find(m_lvlist.begin(), m_lvlist.end(), lv) == m_lvlist.end())
    {
        int lvIdx = m_lvlist.size();  
        int soIdx = lvIdx ; // when converting in postorder soIdx is the same as lvIdx
        m_lvidx[lv] = lvIdx ;  
        m_lvlist.push_back(lv);  

        const G4VSolid* const solid = lv->GetSolid(); 
        const std::string& lvname = lv->GetName() ; 
        const std::string& soname = solid->GetName() ; 

        bool balance_deep_tree = true ;  
        GMesh* mesh = convertSolid( lvIdx, soIdx, solid, lvname, balance_deep_tree ) ;  
        mesh->setIndex( lvIdx ) ;   

        // raw unbalanced tree height  
        const nnode* root = mesh->getRoot(); 
        assert( root ); 
        const nnode* unbalanced = root->other ? root->other : root  ; 
        assert( unbalanced ); 

        unsigned unbalanced_height = unbalanced->maxdepth() ;  
        bool can_export_unbalanced = unbalanced_height <= NCSG::MAX_EXPORT_HEIGHT ;  

        LOG(LEVEL) 
           << " lvIdx " << lvIdx   
           << " soIdx " << soIdx   
           << " unbalanced_height " << unbalanced_height
           << " NCSG::MAX_EXPORT_HEIGHT " << NCSG::MAX_EXPORT_HEIGHT
           << " can_export_unbalanced " << can_export_unbalanced
           ; 
 
        const NCSG* csg = mesh->getCSG(); 
        if( csg->is_balanced() )  // when balancing done, also convert without it 
        {
            if( can_export_unbalanced )
            {  
                balance_deep_tree = false ;  
                GMesh* rawmesh = convertSolid( lvIdx, soIdx, solid, lvname, balance_deep_tree ) ;  
                rawmesh->setIndex( lvIdx ) ;   

                const NCSG* rawcsg = rawmesh->getCSG(); 
                assert( rawmesh->getIndex() == rawcsg->getIndex() ) ;   

                mesh->setAlt(rawmesh);  // <-- this association is preserved (and made symmetric) thru metadata by GMeshLib 
            } 
            else
            {
                LOG(error)
                    << " Cannot export the unbalanced tree as raw height exceeds the maximum. " << std::endl 
                    << " unbalanced_height " << unbalanced_height 
                    << " NCSG::MAX_EXPORT_HEIGHT " << NCSG::MAX_EXPORT_HEIGHT
                    ;     
            } 
        }




        m_lvname.push_back( lvname ); 
        m_soname.push_back( soname ); 
 
        if( root->has_torus() )
        {
            LOG(fatal) << " has_torus lvIdx " << lvIdx << " " << lvname ;  
            m_lv_with_torus.push_back( lvIdx ); 
            m_lvname_with_torus.push_back( lvname ); 
            m_soname_with_torus.push_back( soname ); 
        }

        m_ggeo->add( mesh ) ; 
    }  
}


/**
X4PhysicalVolume::convertSolid
--------------------------------

Converts G4VSolid into two things:

1. analytic CSG nnode tree, boolean solids or polycones convert to trees of multiple nodes,
   deep trees are balanced to reduce their height
2. triangulated vertices and faces held in GMesh instance

As YOG doesnt depend on GGeo, and as workaround for GMesh/GBuffer deficiencies 
the source NPY arrays are also tacked on to the Mh instance.


--x4polyskip 211,232
~~~~~~~~~~~~~~~~~~~~~~

For DYB Near geometry two depth 12 CSG trees needed to be 
skipped as the G4 polygonization goes into an infinite (or at least 
beyond my patience) loop.::

     so:029 lv:211 rmx:12 bmx:04 soName: near_pool_iws_box0xc288ce8
     so:027 lv:232 rmx:12 bmx:04 soName: near_pool_ows_box0xbf8c8a8

Skipping results in placeholder bounding box meshes being
used instead of he real shape. 

**/


GMesh* X4PhysicalVolume::convertSolid( int lvIdx, int soIdx, const G4VSolid* const solid, const std::string& lvname, bool balance_deep_tree ) const 
{
     assert( lvIdx == soIdx );  
     bool dbglv = lvIdx == m_ok->getDbgLV() ; 
     const std::string& soname = solid->GetName() ; 

     LOG(LEVEL)
          << "[ "  
          << lvIdx
          << ( dbglv ? " --dbglv " : "" ) 
          << " soname " << soname
          << " lvname " << lvname
          ;
 
     nnode* raw = X4Solid::Convert(solid, m_ok)  ; 

     if(m_g4codegen) generateTestG4Code(lvIdx, solid, raw); 

     nnode* root = balance_deep_tree ? NTreeProcess<nnode>::Process(raw, soIdx, lvIdx) : raw ;  
     root->other = raw ; 

     const NSceneConfig* config = NULL ; 
     NCSG* csg = NCSG::Adopt( root, config, soIdx, lvIdx );   // Adopt exports nnode tree to m_nodes buffer in NCSG instance
     assert( csg ) ; 
     assert( csg->isUsedGlobally() );
     csg->set_soname( soname.c_str() ) ; 
     csg->set_lvname( lvname.c_str() ) ; 

     bool is_balanced = root != raw ; 
     if(is_balanced) assert( balance_deep_tree == true );  
     csg->set_balanced(is_balanced) ;  

     bool is_x4polyskip = m_ok->isX4PolySkip(lvIdx);   // --x4polyskip 211,232
     if( is_x4polyskip ) LOG(fatal) << " is_x4polyskip " << " soIdx " << soIdx  << " lvIdx " << lvIdx ;  

     GMesh* mesh =  is_x4polyskip ? X4Mesh::Placeholder(solid ) : X4Mesh::Convert(solid ) ; 
     mesh->setCSG( csg ); 

     LOG(LEVEL) << "] " << lvIdx ; 
     return mesh ; 
}



void X4PhysicalVolume::generateTestG4Code( int lvIdx, const G4VSolid* const solid, const nnode* raw) const 
{
     bool dbglv = lvIdx == m_ok->getDbgLV() ; 
     const char* gdmlpath = X4CSG::GenerateTestPath( m_g4codegendir, lvIdx, ".gdml" ) ;  
     bool refs = false ;  
     X4GDMLParser::Write( solid, gdmlpath, refs ); 
     if(dbglv)
     { 
         LOG(info) 
             << ( dbglv ? " --dbglv " : "" ) 
             << "[--g4codegen]"
             << " lvIdx " << lvIdx
             ;
         raw->dump_g4code();  // just for debug 
     }
     X4CSG::GenerateTest( solid, m_ok, m_g4codegendir , lvIdx ) ; 
}













void X4PhysicalVolume::dumpLV() const 
{
   LOG(info)
        << " m_lvidx.size() " << m_lvidx.size() 
        << " m_lvlist.size() " << m_lvlist.size() 
        ;

   assert( m_lvlist.size() == m_lvname.size() );  
   assert( m_lvlist.size() == m_soname.size() );  

   for(unsigned i=0 ; i < m_lvlist.size() ; i++)
   {
       const G4LogicalVolume* lv = m_lvlist[i] ; 

       const std::string& lvn =  lv->GetName() ; 
       assert( strcmp(lvn.c_str(), m_lvname[i].c_str() ) == 0 ); 

       std::cout 
           << " i " << std::setw(5) << i
           << " idx " << std::setw(5) << m_lvidx.at(lv)  
           << " lvname " << std::setw(50) << m_lvname[i]
           << " soname " << std::setw(50) << m_soname[i]
           << std::endl ;  
   }
}

void X4PhysicalVolume::dumpTorusLV() const 
{
    assert( m_lv_with_torus.size() == m_lvname_with_torus.size() ); 
    assert( m_lv_with_torus.size() == m_soname_with_torus.size() ); 
    unsigned num_afflicted = m_lv_with_torus.size() ;  
    if(num_afflicted == 0) return ; 


    LOG(info) << " num_afflicted " << num_afflicted ; 
    std::cout << " lvIdx ( " ; 
    for(unsigned i=0 ; i < num_afflicted ; i++) std::cout << m_lv_with_torus[i] << " " ; 
    std::cout << " ) " << std::endl ;  

    for(unsigned i=0 ; i < num_afflicted ; i++) 
    {
        std::cout 
            << " lv "     << std::setw(4)  << m_lv_with_torus[i] 
            << " lvname " << std::setw(50) << m_lvname_with_torus[i] 
            << " soname " << std::setw(50) << m_soname_with_torus[i] 
            << std::endl 
            ; 
    }

}

std::string X4PhysicalVolume::brief() const 
{
    std::stringstream ss ; 
    ss
        << " selected_node_count " << m_selected_node_count
        << " node_count " << m_selected_node_count
        ;

    return ss.str(); 
}


void X4PhysicalVolume::convertCheck() const 
{
    bool no_nodes = m_selected_node_count == 0 || m_node_count == 0 ; 
    if(no_nodes) 
    {
        LOG(fatal)
            << " NO_NODES ERROR " 
            << brief()
            << std::endl
            << " query " 
            << m_query->desc()
            ;
        assert(0) ; 
    }
}



/**
convertStructure
--------------------

Preorder traverse conversion of the full tree of G4VPhysicalVolume 
into a tree of GVolume, the work mostly done in X4PhysicalVolume::convertNode.
GVolume instances are collected into GGeo/GNodeLib.

Old Notes
~~~~~~~~~~~~

Note that its the YOG model that is updated, that gets
converted to glTF later.  This is done to help keeping 
this code independant of the actual glTF implementation 
used.

* NB this is very similar to the ancient AssimpGGeo::convertStructure, GScene::createVolumeTree

**/


const char* X4PhysicalVolume::TMPDIR = "$TMP/extg4/X4PhysicalVolume" ; 

void X4PhysicalVolume::convertStructure()
{
    assert(m_top) ;
    LOG(info) << "[ creating large tree of GVolume instances" ; 

    const G4VPhysicalVolume* pv = m_top ; 
    GVolume* parent = NULL ; 
    const G4VPhysicalVolume* parent_pv = NULL ; 
    int depth = 0 ;
    bool recursive_select = false ;


    OK_PROFILE("_X4PhysicalVolume::convertStructure");

    m_root = convertStructure_r(pv, parent, depth, parent_pv, recursive_select );

    OK_PROFILE("X4PhysicalVolume::convertStructure");

    convertStructureChecks(); 
}


void X4PhysicalVolume::convertStructureChecks() const 
{
    NTreeProcess<nnode>::SaveBuffer(TMPDIR, "NTreeProcess.npy");      
    NNodeNudger::SaveBuffer(TMPDIR, "NNodeNudger.npy"); 
    X4Transform3D::SaveBuffer(TMPDIR, "X4Transform3D.npy"); 


#ifdef X4_TRANSFORM
    LOG(info) 
        << " m_is_identity0 " << m_is_identity0
        << std::endl 
        << " m_is_identity1 " << m_is_identity1 
        ;
#endif


#ifdef X4_PROFILE    
    LOG(info) 
        << " m_convertNode_dt " << m_convertNode_dt 
        << std::endl 
        << " m_convertNode_boundary_dt " << m_convertNode_boundary_dt 
        << std::endl 
        << " m_convertNode_transformsA_dt " << m_convertNode_transformsA_dt 
        << std::endl 
        << " m_convertNode_transformsB_dt " << m_convertNode_transformsB_dt 
        << std::endl 
        << " m_convertNode_transformsC_dt " << m_convertNode_transformsC_dt 
        << std::endl 
        << " m_convertNode_transformsD_dt " << m_convertNode_transformsD_dt 
        << std::endl 
        << " m_convertNode_transformsE_dt " << m_convertNode_transformsE_dt 
        << std::endl 
        << " m_convertNode_GVolume_dt " << m_convertNode_GVolume_dt 
        ;
#endif
}



/**

X4PhysicalVolume::convertStructure_r
--------------------------------------

Preorder traverse.

**/

GVolume* X4PhysicalVolume::convertStructure_r(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const parent_pv, bool& recursive_select )
{
#ifdef X4_PROFILE
     float t0 = BTimeStamp::RealTime(); 
#endif

     GVolume* volume = convertNode(pv, parent, depth, parent_pv, recursive_select );

#ifdef X4_PROFILE
     float t1 = BTimeStamp::RealTime() ; 
     m_convertNode_dt += t1 - t0 ; 
#endif

     m_ggeo->add(volume); // collect in nodelib

     const G4LogicalVolume* const lv = pv->GetLogicalVolume();
  
     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
     {
         const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
         convertStructure_r(child_pv, volume, depth+1, pv, recursive_select );
     }

     return volume   ; 
}


/**
X4PhysicalVolume::addBoundary
------------------------------

Canonically invoked from X4PhysicalVolume::convertNode during the 
main structural traverse.

For a physical volume and its parent physical volume 
adds(if not already present) a boundary to the GBndLib m_blib instance, 
and returns the index of the newly created or pre-existing boundary.
A boundary is a quadruplet of omat/osur/isur/imat indices.

See notes/issues/ab-blib.rst on getting A-B comparisons to match for boundaries.

**/

unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
{
    const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
    const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;

    const G4Material* const imat_ = lv->GetMaterial() ;
    const G4Material* const omat_ = lv_p ? lv_p->GetMaterial() : imat_ ;  // top omat -> imat 

    const char* omat = X4::BaseName(omat_) ; 
    const char* imat = X4::BaseName(imat_) ; 

    // Why do boundaries with this material pair have surface finding problem for the old route ?
    bool problem_pair  = strcmp(omat, "UnstStainlessSteel") == 0 && strcmp(imat, "BPE") == 0 ; 

    // look for a border surface defined between this and the parent volume, in either direction
    bool first_priority = true ;  
    const G4LogicalSurface* const isur_ = findSurface( pv  , pv_p , first_priority );
    const G4LogicalSurface* const osur_ = findSurface( pv_p, pv   , first_priority );  

    // doubtful of findSurface priority with double skin surfaces, see g4op-

    // the above will not find Opticks SensorSurfaces ... so look for those with GGeo

    /*
    const char* _lv = X4::BaseNameAsis(lv) ;  
    const char* _lv_p = X4::BaseNameAsis(lv_p) ;   // NULL when no lv_p   

    const char* this_name = X4::GDMLName(this) ;
    assert( SStr::HasPointerSuffix(this_name, 12) == true ) ;

    // is Geant4 allocator using placement new into some defined location
    // somehow ?  https://isocpp.org/wiki/faq/dtors#placement-new 
    */

    const char* _lv = X4::GDMLName(lv) ; 
    const char* _lv_p = X4::GDMLName(lv_p) ; 


    bool ps = SStr::HasPointerSuffix(_lv, 6, 12) ;  // 9,12 on macOS 
    if(!ps) LOG(fatal) << " unexpected pointer suffix _lv " << _lv ;  
    assert(ps) ;    

    if( _lv_p )
    {
        bool ps = SStr::HasPointerSuffix(_lv_p, 6, 12); 
        if(!ps) LOG(fatal) << " unexpected pointer suffix _lv " << _lv_p ;  
        assert(ps) ;    
    }

    LOG(debug)
        << " lv names to look for skinsurfaces with "
        << " lv " << lv 
        << " _lv " << _lv
        ;

    LOG(debug)
        << " lv names to look for skinsurfaces with "
        << " lv_p " << lv_p 
        << " _lv_p " << _lv_p
        ;


    const GSkinSurface* g_sslv = m_ggeo->findSkinSurface(_lv) ;  
    const GSkinSurface* g_sslv_p = _lv_p ? m_ggeo->findSkinSurface(_lv_p) : NULL ;  

    if( g_sslv_p )
        LOG(debug) << " node_count " << m_node_count 
                   << " _lv_p   " << _lv_p
                   << " g_sslv_p " << g_sslv_p->getName()
                   ; 


    /*
    int clv = m_ggeo->findCathodeLVIndex( _lv ) ;   // > -1 when found
    int clv_p = m_ggeo->findCathodeLVIndex( _lv_p ) ; 
    assert( clv_p == -1 && "not expecting non-leaf cathode LV " ); 
    bool is_cathode = clv > -1 ; 
    */

    if( problem_pair ) 
        LOG(debug) 
            << " problem_pair "
            << " node_count " << m_node_count 
            << " isur_ " << isur_
            << " osur_ " << osur_
            << " _lv " << _lv 
            << " _lv_p " << _lv_p
            << " g_sslv " << g_sslv
            << " g_sslv_p " << g_sslv_p
            ;



    LOG(debug) 
         << " addBoundary "
         << " omat " << omat 
         << " imat " << imat 
         ;
 
    unsigned boundary = 0 ; 
    if( g_sslv == NULL && g_sslv_p == NULL  )   // no skin surface on this or parent volume, just use bordersurface if there are any
    {
        const char* osur = X4::BaseName( osur_ ); 
        const char* isur = X4::BaseName( isur_ ); 
        boundary = m_blib->addBoundary( omat, osur, isur, imat ); 
    }
    else if( g_sslv && !g_sslv_p )   // skin surface on this volume but not parent : set both osur and isur to this 
    {
        const char* osur = g_sslv->getName(); 
        const char* isur = osur ; 
        boundary = m_blib->addBoundary( omat, osur, isur, imat ); 
    }
    else if( g_sslv_p && !g_sslv )  // skin surface on parent volume but not this : set both osur and isur to this
    {
        const char* osur = g_sslv_p->getName(); 
        const char* isur = osur ; 
        boundary = m_blib->addBoundary( omat, osur, isur, imat ); 
    } 
    else if( g_sslv_p && g_sslv )
    {
        assert( 0 && "fabled double skin found : see notes/issues/ab-blib.rst  " ); 
    }

    return boundary ; 
}



/**
X4PhysicalVolume::convertNode
--------------------------------

* creates pts(GParts) from the csg(NCSG) associated to the mesh(GMesh) for the lvIdx solid. 

* pts(GParts) are associated to the GVolume returned for structural node

* suspect the parallel tree is for gltf creation only ?

* observe the NSensor is always NULL here 

* convertNode is hot code : should move whatever possible elsewhere 


Doing GParts::Make at node level is very repetitive, think 300,000 nodes for JUNO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that boundary name is a node level thing, not mesh level : so are forced to 
do GParts::Make at node level 


Can GParts stuff not needing the boundary be done elsewhere ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actually GPt/GPts may provide a way of avoiding doing GParts::Make here, 
instead the arguments are collected allowing deferred postcache GParts::Create 
from persisted GPts


**/


GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
{
#ifdef X4_PROFILE
    float t00 = BTimeStamp::RealTime(); 
#endif
   
    // record copynumber in GVolume, as thats one way to handle pmtid
    const G4PVPlacement* placement = dynamic_cast<const G4PVPlacement*>(pv); 
    assert(placement); 
    G4int copyNumber = placement->GetCopyNo() ;  

    X4Nd* parent_nd = parent ? static_cast<X4Nd*>(parent->getParallelNode()) : NULL ;

    unsigned boundary = addBoundary( pv, pv_p );
    std::string boundaryName = m_blib->shortname(boundary); 
    int materialIdx = m_blib->getInnerMaterial(boundary); 


    const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
    const std::string& lvName = lv->GetName() ; 
    const std::string& pvName = pv->GetName() ; 
    unsigned ndIdx = m_node_count ;       // incremented below after GVolume instanciation

    int lvIdx = m_lvidx[lv] ;   // from postorder traverse in convertSolids to match GDML lvIdx : mesh identity uses lvIdx

    LOG(verbose) 
        << " copyNumber " << std::setw(8) << copyNumber
        << " boundary " << std::setw(4) << boundary 
        << " materialIdx " << std::setw(4) << materialIdx
        << " boundaryName " << boundaryName
        << " lvIdx " << lvIdx
        ;

    // THIS IS HOT NODE CODE : ~300,000 TIMES FOR JUNO 


    const GMesh* mesh = m_hlib->getMeshWithIndex(lvIdx); 

    const NCSG* csg = mesh->getCSG();  
    unsigned csgIdx = csg->getIndex() ; 


#ifdef GPARTS_OLD
    GParts* parts = GParts::Make( csg, boundaryName.c_str(), ndIdx );  // painful to do this here in hot node code
    parts->setBndLib(m_blib);
    //parts->setVolumeIndex( ndIdx );  
    unsigned volIdx = parts->getVolumeIndex(0); 
    assert( volIdx == ndIdx ); 
#endif


     ///////////////////////////////////////////////////////////////  

#ifdef X4_PROFILE
    float t10 = BTimeStamp::RealTime(); 
#endif

    GPt* pt = new GPt( lvIdx, ndIdx, csgIdx, boundaryName.c_str() )  ; 

    glm::mat4 xf_local_t = X4Transform3D::GetObjectTransform(pv);  

#ifdef X4_TRANSFORM
    // check the Object and Frame transforms are inverses of each other
    glm::mat4 xf_local_v = X4Transform3D::GetFrameTransform(pv);  
    glm::mat4 id0 = xf_local_t * xf_local_v ; 
    glm::mat4 id1 = xf_local_v * xf_local_t ; 
    bool is_identity0 = nglmext::is_identity(id0)  ; 
    bool is_identity1 = nglmext::is_identity(id1)  ; 

    m_is_identity0 += ( is_identity0 ? 1 : 0 ); 
    m_is_identity1 += ( is_identity1 ? 1 : 0 ); 

    if(ndIdx < 10 || !(is_identity0 && is_identity1))
    {
        LOG(info) 
            << " ndIdx  " << ndIdx 
            << " is_identity0 " << is_identity0
            << " is_identity1 " << is_identity1
            << std::endl
            << " id0 " << gformat(id0)
            << std::endl
            << " id1 " << gformat(id1)
            << std::endl
            << " xf_local_t " << gformat(xf_local_t)
            << std::endl
            << " xf_local_v " << gformat(xf_local_v)
            ;       
    }
    //assert( is_identity ) ;  
#endif


#ifdef X4_PROFILE
    float t12 = BTimeStamp::RealTime(); 
#endif

    const nmat4triple* ltriple = m_xform->make_triple( glm::value_ptr(xf_local_t) ) ;   // YIKES does polardecomposition + inversion and checks them 
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#ifdef X4_PROFILE
    float t13 = BTimeStamp::RealTime(); 
#endif

    GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local_t));

#ifdef X4_PROFILE
    float t15 = BTimeStamp::RealTime(); 
#endif

    X4Nd* nd = new X4Nd { parent_nd, ltriple } ;        

    const nmat4triple* gtriple = nxform<X4Nd>::make_global_transform(nd) ; 
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#ifdef X4_PROFILE
    float t17 = BTimeStamp::RealTime(); 
#endif

    glm::mat4 xf_global = gtriple->t ;

    GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));

#ifdef X4_PROFILE
    float t20 = BTimeStamp::RealTime(); 

    m_convertNode_boundary_dt    += t10 - t00 ; 

    m_convertNode_transformsA_dt += t12 - t10 ; 
    m_convertNode_transformsB_dt += t13 - t12 ; 
    m_convertNode_transformsC_dt += t15 - t13 ; 
    m_convertNode_transformsD_dt += t17 - t15 ; 
    m_convertNode_transformsE_dt += t20 - t17 ; 
#endif

/*
     m_convertNode_boundary_dt 3.47852
     m_convertNode_transformsA_dt 0.644531
     m_convertNode_transformsB_dt 6.35547
     m_convertNode_transformsC_dt 0.277344
     m_convertNode_transformsD_dt 7.29492
     m_convertNode_transformsE_dt 0.230469
     m_convertNode_GVolume_dt 3
*/

    ////////////////////////////////////////////////////////////////

    G4PVPlacement* _placement = const_cast<G4PVPlacement*>(placement) ;  
    void* origin_node = static_cast<void*>(_placement) ; 
    GVolume* volume = new GVolume(ndIdx, gtransform, mesh, origin_node );
    m_node_count += 1 ; 

    unsigned lvr_lvIdx = lvIdx ; 
    bool selected = m_query->selected(pvName.c_str(), ndIdx, depth, recursive_select, lvr_lvIdx );
    if(selected) m_selected_node_count += 1 ;  

    LOG(verbose) << " lv_lvIdx " << lvr_lvIdx
                 << " selected " << selected
                 ; 

    int sensorIndex = m_blib->isSensorBoundary(boundary) ? m_ggeo->addSensorVolume(volume) : -1 ; 
    if(sensorIndex > -1) m_blib->countSensorBoundary(boundary); 

    /*
    if(sensorIndex > -1)
    {
        LOG(info)
            << " copyNumber " << std::setw(8) << copyNumber
            << " sensorIndex " << std::setw(8) << sensorIndex
            << " boundary " << std::setw(4) << boundary 
            << " boundaryName " << boundaryName
            ;
    }
    */
 
#ifdef OLD_SENSOR
    NSensor* sensor = NULL ; 
    volume->setSensor( sensor );   
#endif
    volume->setSensorIndex(sensorIndex); 

    volume->setCopyNumber(copyNumber);  // NB within instances this is changed by GInstancer::labelRepeats_r when m_duplicate_outernode_copynumber is true
    volume->setBoundary( boundary ); 
    volume->setSelected( selected );

    volume->setLevelTransform(ltransform);

    volume->setLocalTransform(ltriple);
    volume->setGlobalTransform(gtriple);
 
    volume->setParallelNode( nd ); 

#ifdef GPARTS_HOT
     volume->setParts( parts ); 
#endif

    volume->setPt( pt ); 
    volume->setPVName( pvName.c_str() );
    volume->setLVName( lvName.c_str() );
    volume->setName( pvName.c_str() );   // historically (AssimpGGeo) this was set to lvName, but pvName makes more sense for node node

    m_ggeo->countMeshUsage(lvIdx, ndIdx );

    if(parent) 
    {
         parent->addChild(volume);
         volume->setParent(parent);
    } 


#ifdef X4_PROFILE
    float t30 = BTimeStamp::RealTime() ;
    m_convertNode_GVolume_dt     += t30 - t20 ; 
#endif


    return volume ; 
}


void X4PhysicalVolume::DumpSensorVolumes(const GGeo* gg, const char* msg)
{
    unsigned numSensorVolumes = gg->getNumSensorVolumes();
    LOG(info) 
         << msg
         << " numSensorVolumes " << numSensorVolumes 
         ; 


    unsigned lastTransitionIndex = -2 ; 
    unsigned lastOuterCopyNo = -2 ; 

    for(unsigned i=0 ; i < numSensorVolumes ; i++)
    {   
        unsigned sensorIdx = i ; 

        const GVolume* sensor = gg->getSensorVolume(sensorIdx) ; 
        assert(sensor); 
        const void* const sensorOrigin = sensor->getOriginNode(); 
        assert(sensorOrigin);
        const G4PVPlacement* const sensorPlacement = static_cast<const G4PVPlacement* const>(sensorOrigin);
        assert(sensorPlacement);  
        unsigned sensorCopyNo = sensorPlacement->GetCopyNo() ;  


        const GVolume* outer = sensor->getOuterVolume() ; 
        assert(outer); 
        const void* const outerOrigin = outer->getOriginNode(); 
        assert(outerOrigin); 
        const G4PVPlacement* const outerPlacement = static_cast<const G4PVPlacement* const>(outerOrigin);
        assert(outerPlacement);
        unsigned outerCopyNo = outerPlacement->GetCopyNo() ;  

        if(outerCopyNo != lastOuterCopyNo + 1) lastTransitionIndex = i ; 
        lastOuterCopyNo = outerCopyNo ; 

        if(i - lastTransitionIndex < 10)
        std::cout 
             << " sensorIdx " << std::setw(6) << sensorIdx 
             << " sensorPlacement " << std::setw(8) << sensorPlacement
             << " sensorCopyNo " << std::setw(8) << sensorCopyNo
             << " outerPlacement " << std::setw(8) << outerPlacement
             << " outerCopyNo " << std::setw(8) << outerCopyNo
             << std::endl 
             ;
    }
}

void X4PhysicalVolume::GetSensorPlacements(const GGeo* gg, std::vector<G4PVPlacement*>& placements) // static
{
    placements.clear(); 

    std::vector<void*> placements_ ; 
    gg->getSensorPlacements(placements_); 

    for(unsigned i=0 ; i < placements_.size() ; i++)
    {
         G4PVPlacement* p = static_cast<G4PVPlacement*>(placements_[i]); 
         placements.push_back(p); 
    } 
}


void X4PhysicalVolume::DumpSensorPlacements(const GGeo* gg, const char* msg) // static
{
    std::vector<G4PVPlacement*> sensors ; 
    X4PhysicalVolume::GetSensorPlacements(gg, sensors);
    int num_sen = sensors.size();  

    LOG(info) << msg <<  " num_sen " << num_sen ; 

    int lastCopyNo = -2 ;   
    int lastTransition = -2 ; 
    int margin = 10 ; 

    for(int i=0 ; i < num_sen ; i++)
    {   
         int sensorIdx = i ; 
         const G4PVPlacement* sensor = sensors[sensorIdx] ; 
         G4int copyNo = sensor->GetCopyNo(); 

         if( lastCopyNo + 1 != copyNo ) lastTransition = i ; 

         if( i - lastTransition < margin || i < margin || num_sen - 1 - i < margin ) 
         std::cout 
             << " sensorIdx " << std::setw(6) << sensorIdx
             << " sensor " << std::setw(8) << sensor
             << " copyNo " << std::setw(6) << copyNo
             << std::endl 
             ;   

         lastCopyNo = copyNo ; 
    }   
}




