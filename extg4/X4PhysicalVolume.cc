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


#include "GMesh.hh"
#include "GGeo.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"

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
    pv.saveAsGLTF("/tmp/X4PhysicalVolume/X4PhysicalVolume.gltf"); // TODO: should be using identity digest in the path  
    GGeo* gg = pv.getGGeo();
    return gg ; 
}

X4PhysicalVolume::X4PhysicalVolume(const G4VPhysicalVolume* const top)
    :
    m_top(top),
    m_key(Key(m_top)),
    m_keyset(BOpticksKey::SetKey(m_key)),
    m_ok(Opticks::GetOpticks()),  // Opticks instanciation must be after BOpticksKey::SetKey
    m_ggeo(new GGeo(m_ok)),
    m_mlib(m_ggeo->getMaterialLib()),
    m_slib(m_ggeo->getSurfaceLib()),
    m_blib(m_ggeo->getBndLib()),
    m_sc(new YOG::Sc(0)),
    m_maker(new YOG::Maker(m_sc)),
    m_verbosity(m_ok->getVerbosity()),
    m_pvcount(0),
    m_identity()
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
}

void X4PhysicalVolume::convertStructure()
{
     assert(m_top) ;
     LOG(info) << " sc BEGIN " << m_sc->desc() ; 

     const G4VPhysicalVolume* pv = m_top ; 
     const G4VPhysicalVolume* parent_pv = NULL ; 
     int depth = 0 ;
     int preorder = 0 ; 

     IndexTraverse(pv, depth);
     TraverseVolumeTree(pv, depth, preorder, parent_pv );

     LOG(info) << " sc END  " << m_sc->desc() ; 
}

void X4PhysicalVolume::saveAsGLTF(const char* path)
{
     m_maker->convert();
     m_maker->save(path);
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

**/

int X4PhysicalVolume::TraverseVolumeTree(const G4VPhysicalVolume* const pv, int depth, int preorder, const G4VPhysicalVolume* const parent_pv )
{
     const G4LogicalVolume* const lv = pv->GetLogicalVolume() ;

     Nd* nd = convertNodeVisit(pv, depth, parent_pv );
     preorder += 1 ; 
  
     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
     {
         const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
         int child_ndIdx = TraverseVolumeTree(child_pv,depth+1, preorder,  pv );
         nd->children.push_back(child_ndIdx); 
     }

     return nd->ndIdx  ; 
}

/**
X4PhysicalVolume::convertNodeVisit
------------------------------------

* cf AssimpGGeo::convertStructureVisit which returns GSolid (recall thats poorly named actually its "node" like)

**/



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


Nd* X4PhysicalVolume::convertNodeVisit(const G4VPhysicalVolume* const pv, int depth, const G4VPhysicalVolume* const pv_p )
{

     G4RotationMatrix pv_rotation = pv->GetObjectRotationValue() ;  // obj relative to mother
     G4ThreeVector    pv_translation = pv->GetObjectTranslation() ;
     G4Transform3D    pv_transform(pv_rotation,pv_translation);

     glm::mat4* transform = new glm::mat4(X4Transform3D::Convert( pv_transform ));

     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
     const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;

     const G4Material* const imat = lv->GetMaterial() ;
     const G4Material* const omat = lv_p ? lv_p->GetMaterial() : imat ;   
     // treat parent of world as same material as world

     bool first_priority = true ;  
     const G4LogicalSurface* const isur = findSurface( pv  , pv_p , first_priority );
     const G4LogicalSurface* const osur = findSurface( pv_p, pv   , first_priority );  
     // doubtful of this, see g4op-

    /*
     GBndLib::addBoundary requires getting the indices for the materials
     and surfaces, but that triggers closure of the libs... which should
     be done only when the traverse is complete.  

     Hmm, perhaps do an initial traverse to collect all 
     materials and surfaces, then close mlib and slib before 
     this traverse.   

     Actually do not need to traverse to find all materials
     and surfaces just convert them from their stores.  Having more
     than are used is not a problem.

     unsigned boundary_ = m_blib->addBoundary( 
                                                X4::ShortName(omat),  
                                                X4::ShortName(osur),                   
                                                X4::ShortName(isur),  
                                                X4::ShortName(imat)       
                                            );   
     */
     std::string boundary = BStr::join(
                                        X4::ShortName(omat), 
                                        X4::ShortName(osur),
                                        X4::ShortName(isur),
                                        X4::ShortName(imat),
                                        '/'
                                       ) ;


     int materialIdx = convertMaterialVisit( imat );
 
     G4VSolid* solid = lv->GetSolid();

     int lvIdx = m_lvidx[lv] ;  // from a prior postorder IndexTraverse, to match the lvIdx obtained from GDML 
     const std::string& lvName = lv->GetName() ;
     const std::string& pvName = pv->GetName() ; 
     const std::string& soName = solid->GetName() ; 
     bool selected  = true ; 

     int ndIdx = m_sc->add_node(
                                 lvIdx, 
                                 materialIdx,
                                 lvName,
                                 pvName,
                                 soName,
                                 transform,
                                 boundary,
                                 depth,
                                 selected
                               );

     Nd* nd = m_sc->get_node(ndIdx) ; 
     Mh* mh = m_sc->get_mesh_for_node( ndIdx ); 
     if(mh->csg == NULL) convertSolid(mh, solid);

     return nd ; 
}

/**
convertMaterialVisit
----------------------

Recall the complication with material indices, need to 
rearrange to make important materials have low indices
for on-GPU step-by-step material tracing using small(4?) 
numbers of bits to record the material 

When does this reordering happen ?

* Is that the source of the material lookups ?
* There are also lookups into the boundary texture lines.
**/

int X4PhysicalVolume::convertMaterialVisit(const G4Material* const material )
{
    const std::string& matname_ = material->GetName(); 
    const char* matname = matname_.c_str() ; 

    int materialIdx = m_sc->add_material(matname_) ;  // only adds for new material names
    if(!m_mlib->hasMaterial(matname))
    { 
        GMaterial* mat = X4Material::Convert(material) ; 
        int mlibIdx = m_mlib->getNumMaterials();
        mat->setIndex( mlibIdx ); 
        m_ggeo->add(mat);  // huh: ggeo/mlib ? 
        assert( materialIdx == mlibIdx );   
    }
    return materialIdx ; 
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

void X4PhysicalVolume::convertSolid( Mh* mh,  G4VSolid* solid)
{
     mh->csg = X4Solid::Convert(solid) ;   

     GMesh* mesh = X4Mesh::Convert(solid) ; 
     mh->mesh = mesh ; 
     mh->vtx = mesh->m_x4src_vtx ; 
     mh->idx = mesh->m_x4src_idx ; 
} 

