#include <iostream>
#include <sstream>
#include <iomanip>

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4Material.hh"
#include "G4VSolid.hh"
#include "G4TransportationManager.hh"

#include "X4.hh"
#include "X4PhysicalVolume.hh"
#include "X4Material.hh"
#include "X4MaterialTable.hh"
#include "X4LogicalBorderSurfaceTable.hh"
#include "X4Solid.hh"
#include "X4Mesh.hh"
#include "X4Transform3D.hh"


#include "YOG.hh"
#include "YOGMaker.hh"

using YOG::Sc ; 
using YOG::Nd ; 
using YOG::Mh ; 
using YOG::Maker ; 


class NSensor ; 

#include "GMesh.hh"
#include "GVolume.hh"
#include "GGeo.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"

#include "NXform.hpp"
#include "BStr.hh"
#include "BOpticksKey.hh"
#include "Opticks.hh"
#include "SDigest.hh"
#include "PLOG.hh"


const G4VPhysicalVolume* const X4PhysicalVolume::Top()
{
    const G4VPhysicalVolume* const top = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;
    return top ; 
}

GGeo* X4PhysicalVolume::Convert(const G4VPhysicalVolume* const top)
{
    X4PhysicalVolume pv(top) ;  

    //const char* path = "/tmp/X4PhysicalVolume/X4PhysicalVolume.gltf" ;
    const char* path = NULL ; 

    pv.saveAsGLTF(path); 

    GGeo* gg = pv.getGGeo();
    return gg ; 
}

X4PhysicalVolume::X4PhysicalVolume(const G4VPhysicalVolume* const top)
    :
    m_top(top),
    m_key(Key(m_top)),
    m_keyset(BOpticksKey::SetKey(m_key)),
    m_ok(Opticks::GetOpticks()),  // Opticks instanciation must be after BOpticksKey::SetKey
    m_gltfpath(m_ok->getGLTFPath()),
    m_ggeo(new GGeo(m_ok)),
    m_mlib(m_ggeo->getMaterialLib()),
    m_slib(m_ggeo->getSurfaceLib()),
    m_blib(m_ggeo->getBndLib()),
    m_xform(new nxform(0,false)),
    m_sc(new YOG::Sc(0)),
    m_maker(new YOG::Maker(m_sc)),
    m_verbosity(m_ok->getVerbosity())
{
    init();
}

GGeo* X4PhysicalVolume::getGGeo()
{
    return m_ggeo ; 
}

void X4PhysicalVolume::init()
{
    convertMaterials();
    convertSurfaces();
    convertStructure();
}


void X4PhysicalVolume::convertMaterials()
{
    size_t num_materials0 = m_mlib->getNumMaterials() ;
    assert( num_materials0 == 0 );

    X4MaterialTable::Convert(m_mlib);

    size_t num_materials = m_mlib->getNumMaterials() ;
    assert( num_materials > 0 );

    LOG(info) << "convertMaterials"
              << " num_materials " << num_materials
              ;

    m_mlib->close();   // may change order if prefs dictate
}

void X4PhysicalVolume::convertSurfaces()
{
    size_t num_surf0 = m_slib->getNumSurfaces() ; 
    assert( num_surf0 == 0 );

    X4LogicalBorderSurfaceTable::Convert(m_slib);

    size_t num_lbs = m_slib->getNumSurfaces() ; 

    //X4LogicalSkinSurfaceTable::Convert(m_slib);
    size_t num_sks = m_slib->getNumSurfaces() - num_lbs ; 

    LOG(info) << "convertSurfaces"
              << " num_lbs " << num_lbs
              << " num_sks " << num_sks
              ;
    m_slib->close();  // may change order if prefs dictate
}


void X4PhysicalVolume::saveAsGLTF(const char* path)
{
     m_maker->convert();
     m_maker->save(path ? path : m_gltfpath);
}

std::string X4PhysicalVolume::Digest( const G4LogicalVolume* const lv, const G4int depth )
{
    SDigest dig ;

    for (unsigned i=0; i < unsigned(lv->GetNoDaughters()) ; i++)
    {
        const G4VPhysicalVolume* const d_pv = lv->GetDaughter(i);

        G4RotationMatrix rot, invrot;

        if (d_pv->GetFrameRotation() != 0)
        {
           rot = *(d_pv->GetFrameRotation());
           invrot = rot.inverse();
        }

        std::string d_dig = Digest(d_pv->GetLogicalVolume(),depth+1);

        // postorder visit region is here after the recursive call

        G4Transform3D P(invrot,d_pv->GetObjectTranslation());

        std::string p_dig = X4Transform3D::Digest(P) ; 
    
        dig.update( const_cast<char*>(d_dig.data()), d_dig.size() );  
        dig.update( const_cast<char*>(p_dig.data()), p_dig.size() );  
    }

    // Avoid pointless repetition of full material digests for every 
    // volume by digesting just the material name (could use index instead)
    // within the recursion.
    //
    // Full material digests of all properties are included after the recursion.

    G4Material* material = lv->GetMaterial();
    const G4String& name = material->GetName();    
    dig.update( const_cast<char*>(name.data()), name.size() );  

    return dig.finalize();
}


std::string X4PhysicalVolume::Digest( const G4VPhysicalVolume* const top)
{
    const G4LogicalVolume* lv = top->GetLogicalVolume() ;
    std::string tree = Digest(lv, 0 ); 
    std::string mats = X4Material::Digest(); 

    SDigest dig ;
    dig.update( const_cast<char*>(tree.data()), tree.size() );  
    dig.update( const_cast<char*>(mats.data()), mats.size() );  
    return dig.finalize();
}


const char* X4PhysicalVolume::Key(const G4VPhysicalVolume* const top )
{
    std::string digest = Digest(top);

    const char* exename = PLOG::instance->args.exename() ; 

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


void X4PhysicalVolume::IndexTraverse(const G4VPhysicalVolume* const pv, int depth)
{
    const G4LogicalVolume* const lv = pv->GetLogicalVolume() ;
    for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    {
        const G4VPhysicalVolume* const daughter_pv = lv->GetDaughter(i);
        IndexTraverse( daughter_pv , depth + 1 );
    }
    // record the tail/postorder idx for the lv
    m_lvidx[lv] = m_lvidx.size(); 
}



G4LogicalSurface* X4PhysicalVolume::findSurface( const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_priority )
{
     G4LogicalSurface* surf = NULL ; 

     surf = G4LogicalBorderSurface::GetSurface(a, b) ;

     const G4VPhysicalVolume* const first  = first_priority ? a : b ; 
     const G4VPhysicalVolume* const second = first_priority ? b : a ; 

     if(surf == NULL)
         surf = G4LogicalSkinSurface::GetSurface(first ? first->GetLogicalVolume() : NULL );

     if(surf == NULL)
         surf = G4LogicalSkinSurface::GetSurface(second ? second->GetLogicalVolume() : NULL );

     return surf ; 
}





/**
TraverseVolumeTree
--------------------

Moving convertNode to postorder position in the tail, 
would avoid the separate IndexTraverse to give m_lvidx
BUT preorder node indices (root being zero) are nicer, and would have to 
collect vectors of child indices.

The reason to keep using postorder indices is to 
match GDML lvIdx.

Note that its the YOG model that is updated, that gets
converted to glTF later.  This is done to help keeping 
this code independant of the actual glTF implementation 
used.

TODO: contrast with AssimpGGeo::convertStructure

* need to create GVolume(GNode) and hookem up into a tree here, but 
  YOG::Nd is also needed 

  * maybe add a void* "aux" slot to GNode to passively hold the YOG::Nd  
    then can return GVolume(GNode) in the traverse but still have the 
    much simpler YOG::Nd to work with YOG::Maker
 

**/


void X4PhysicalVolume::convertStructure()
{
     assert(m_top) ;
     LOG(info) << " convertStructure BEGIN " << m_sc->desc() ; 

     const G4VPhysicalVolume* pv = m_top ; 
     GVolume* parent = NULL ; 
     const G4VPhysicalVolume* parent_pv = NULL ; 
     int depth = 0 ;
     int preorder = 0 ; 

     IndexTraverse(pv, depth);

     m_root = convertTree_r(pv, parent, depth, preorder, parent_pv );

     LOG(info) << " convertStructure END  " << m_sc->desc() ; 
}


GVolume* X4PhysicalVolume::convertTree_r(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, int preorder, const G4VPhysicalVolume* const parent_pv )
{
     GVolume* volume = convertNode(pv, depth, preorder, parent_pv );

     YOG::Nd* nd = static_cast<YOG::Nd*>(volume->getParallelNode()) ;
     YOG::Nd* parent_nd = parent ? static_cast<YOG::Nd*>(parent->getParallelNode()) : NULL ;

     if(parent) 
     {
         parent->addChild(volume);
         volume->setParent(parent);
     } 

     if(parent_nd)
     {
         parent_nd->children.push_back(nd->ndIdx);
     }

     // TODO: need parent hookup, somewhere here, then can form the global


     const G4LogicalVolume* const lv = pv->GetLogicalVolume();

     preorder += 1 ; 
  
     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
     {
         const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
         convertTree_r(child_pv, volume, depth+1, preorder, pv );
     }

     return volume   ; 
}

/**
X4PhysicalVolume::convertNodeVisit
------------------------------------
**/

GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, int depth, int preorder, const G4VPhysicalVolume* const pv_p )
{
     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
     const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;

     const G4Material* const imat = lv->GetMaterial() ;
     const G4Material* const omat = lv_p ? lv_p->GetMaterial() : imat ;   
     // treat parent of world as same material as world

     bool first_priority = true ;  
     const G4LogicalSurface* const isur = findSurface( pv  , pv_p , first_priority );
     const G4LogicalSurface* const osur = findSurface( pv_p, pv   , first_priority );  
     // doubtful of findSurface priority with double skin surfaces, see g4op-

     unsigned boundary = m_blib->addBoundary( 
                                                X4::ShortName(omat),  
                                                X4::ShortName(osur),                   
                                                X4::ShortName(isur),  
                                                X4::ShortName(imat)       
                                            );
     std::string boundaryName = m_blib->shortname(boundary); 
     guint4 bnd = m_blib->getBnd(boundary); 
     int materialIdx = m_mlib->getIndex(X4::ShortName(imat)) ;
     assert( materialIdx == bnd.w ); 


     // from a prior postorder IndexTraverse, to match the lvIdx obtained from GDML 
     // mesh identity is based on lvIdx 
     int lvIdx = m_lvidx[lv] ;  

     const G4VSolid* const solid = lv->GetSolid();
     const std::string& lvName = lv->GetName() ;
     const std::string& pvName = pv->GetName() ; 
     const std::string& soName = solid->GetName() ; 
     bool selected  = true ; 

     glm::mat4* xf_local = X4Transform3D::GetLocalTransform(pv); 
     const nmat4triple* transform = m_xform->make_triple( glm::value_ptr(*xf_local) ) ; 

     int ndIdx = m_sc->add_node(
                                 lvIdx, 
                                 materialIdx,
                                 lvName,
                                 pvName,
                                 soName,
                                 transform,
                                 boundaryName,
                                 depth,
                                 selected
                               );

     assert( ndIdx == preorder ); 

     Nd* nd = m_sc->get_node(ndIdx) ; 
     Mh* mh = m_sc->get_mesh_for_node( ndIdx );  // node->mesh via soIdx (the local mesh index)

     if(mh->csg == NULL) convertSolid(mh, solid);

     GMesh* mesh = mh->mesh ;  
     // CAUTION : AssimpGGeo::convertMeshes does some mesh processing (deduping, fixing) 
     // before inclusion in the GVolume(GNode) 

     GParts* pts = NULL ;   // TODO: get GParts setup from recursive vistor GScene::createVolume 
     NSensor* sensor = NULL ; 

     glm::mat4 xf_global ; // placeholder, until have parent hookup so can get the global
     GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));
     GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(*xf_local));

     GVolume* volume = new GVolume(ndIdx, gtransform, mesh, boundary, sensor );

     volume->setLevelTransform(ltransform);

     volume->setParallelNode( nd ); 
     volume->setParts( pts ); 

     return volume ; 
}


/**
convertSolid
-------------

Converts G4VSolid into two things:

1. analytic CSG nnode tree, boolean solids or polycones convert to trees of multiple nodes
2. triangulated vertices and faces held in GMesh instance

As YOG doesnt depend on GGeo, and as workaround for GMesh/GBuffer deficiencies 
the source NPY arrays are also tacked on to the Mh instance.

**/

void X4PhysicalVolume::convertSolid( Mh* mh, const G4VSolid* const solid)
{
     mh->csg = X4Solid::Convert(solid) ;   

     GMesh* mesh = X4Mesh::Convert(solid) ; 
     mh->mesh = mesh ; 
     mh->vtx = mesh->m_x4src_vtx ; 
     mh->idx = mesh->m_x4src_idx ; 
} 

