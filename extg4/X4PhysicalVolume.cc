#include <iostream>
#include <sstream>
#include <iomanip>

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4Material.hh"
#include "G4VisExtent.hh"
#include "G4VSolid.hh"
#include "G4TransportationManager.hh"

#include "X4.hh"
#include "X4PhysicalVolume.hh"
#include "X4Material.hh"
#include "X4MaterialTable.hh"
#include "X4LogicalBorderSurfaceTable.hh"
#include "X4LogicalSkinSurfaceTable.hh"
#include "X4Solid.hh"
#include "X4CSG.hh"
#include "X4Mesh.hh"
#include "X4Transform3D.hh"

#include "YOG.hh"
#include "YOGMaker.hh"

using YOG::Sc ; 
using YOG::Nd ; 
using YOG::Mh ; 
using YOG::Maker ; 

#include "SSys.hh"
#include "SDigest.hh"
#include "PLOG.hh"
#include "BStr.hh"
#include "BFile.hh"
#include "BOpticksKey.hh"

class NSensor ; 

#include "NXform.hpp"  // header with the implementation
template struct nxform<YOG::Nd> ; 

#include "NGLMExt.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NNodeNudger.hpp"
#include "NTreeProcess.hpp"

#include "GMesh.hh"
#include "GVolume.hh"
#include "GParts.hh"
#include "GGeo.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"

#include "Opticks.hh"
#include "OpticksQuery.hh"



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

    //Opticks* ok = Opticks::GetOpticks() ; 

    Opticks* ok = new Opticks(0,0);  // Opticks instanciation must be after BOpticksKey::SetKey

    GGeo* gg = new GGeo(ok) ;

    X4PhysicalVolume xtop(gg, top) ;  

    const char* path = NULL ; 
    int root = 0 ; 

    xtop.saveAsGLTF(root, path);  // <-- TODO: give GGeo this capability 
 
    return gg ; 
}




X4PhysicalVolume::X4PhysicalVolume(GGeo* ggeo, const G4VPhysicalVolume* const top)
    :
    m_ggeo(ggeo),
    m_top(top),
    m_ok(m_ggeo->getOpticks()), 
    m_query(m_ok->getQuery()),
    m_gltfpath(m_ok->getGLTFPath()),
    m_g4codegen(m_ok->isG4CodeGen()),
    m_g4codegendir(m_ok->getG4CodeGenDir()),
    m_mlib(m_ggeo->getMaterialLib()),
    m_slib(m_ggeo->getSurfaceLib()),
    m_blib(m_ggeo->getBndLib()),
    m_xform(new nxform<YOG::Nd>(0,false)),
    m_sc(new YOG::Sc(0)),
    m_maker(new YOG::Maker(m_sc)),
    m_verbosity(m_ok->getVerbosity()),
    m_ndCount(0)
{
    const char* msg = "GGeo ctor argument of X4PhysicalVolume must have mlib, slib and blib already " ; 
    assert( m_mlib && msg ); 
    assert( m_slib && msg ); 
    assert( m_blib && msg ); 

    init();
}

GGeo* X4PhysicalVolume::getGGeo()
{
    return m_ggeo ; 
}

void X4PhysicalVolume::init()
{
    LOG(info) << "query : " << m_query->desc() ; 

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


    // TODO : can these go into one method within GMaterialLib?
    m_mlib->addTestMaterials() ;

    m_mlib->close();   // may change order if prefs dictate

    // replaceGROUPVE needs the buffer : so must be after close
    bool debug = false ; 
    m_mlib->replaceGROUPVEL(debug); 

    // TODO: do the GROUPVEL calc be directly within Geant4, to avoid the kludging 

    // getting names must be done after the close

    for(size_t i=0 ; i < num_materials ; i++)
    {
        const char* name = m_mlib->getName(i); 
        int idx = m_sc->add_material(name); 
        assert( idx == int(i) ); 

        if(m_verbosity > 5)
            std::cout 
                << std::setw(4) << i 
                << " : " 
                << name
                << std::endl 
                ;
        }

}

void X4PhysicalVolume::convertSurfaces()
{
    size_t num_surf0 = m_slib->getNumSurfaces() ; 
    assert( num_surf0 == 0 );

    X4LogicalBorderSurfaceTable::Convert(m_slib);
    size_t num_lbs = m_slib->getNumSurfaces() ; 

    X4LogicalSkinSurfaceTable::Convert(m_slib);
    size_t num_sks = m_slib->getNumSurfaces() - num_lbs ; 

    LOG(info) << "convertSurfaces"
              << " num_lbs " << num_lbs
              << " num_sks " << num_sks
              ;

    m_slib->addPerfectSurfaces();

    m_slib->close();  // may change order if prefs dictate
}


void X4PhysicalVolume::saveAsGLTF(int root, const char* path)
{
     m_sc->root = SSys::getenvint("GLTF_ROOT", root);  // 3147
     LOG(error) << "X4PhysicalVolume::saveAsGLTF"
                << " sc.root " << m_sc->root 
                ;

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
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    {
        const G4VPhysicalVolume* const daughter_pv = lv->GetDaughter(i);
        IndexTraverse( daughter_pv , depth + 1 );
    }

    // for newly encountered lv record the tail/postorder idx for the lv
    if(std::find(m_lvlist.begin(), m_lvlist.end(), lv) == m_lvlist.end())
    {
        m_lvidx[lv] = m_lvlist.size(); 
        m_lvlist.push_back(lv);  
    }  
}

void X4PhysicalVolume::dumpLV()
{
   LOG(info)
        << " m_lvidx.size() " << m_lvidx.size() 
        << " m_lvlist.size() " << m_lvlist.size() 
        ;

   for(unsigned i=0 ; i < m_lvlist.size() ; i++)
   {
       const G4LogicalVolume* lv = m_lvlist[i] ; 
       std::cout 
           << " i " << std::setw(5) << i
           << " idx " << std::setw(5) << m_lvidx[lv]  
           << " lv "  << lv->GetName()
           << std::endl ;  
   }
}

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
convertStructure
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

* NB this is very similar to AssimpGGeo::convertStructure, GScene::createVolumeTree

**/

void X4PhysicalVolume::convertStructure()
{
     assert(m_top) ;
     LOG(info) << " convertStructure BEGIN " << m_sc->desc() ; 

     const G4VPhysicalVolume* pv = m_top ; 
     GVolume* parent = NULL ; 
     const G4VPhysicalVolume* parent_pv = NULL ; 
     int depth = 0 ;

     IndexTraverse(pv, depth);

     if(m_verbosity > 5) dumpLV();

     bool recursive_select = false ;

     m_root = convertTree_r(pv, parent, depth, parent_pv, recursive_select );

     NTreeProcess<nnode>::SaveBuffer("$TMP/NTreeProcess.npy");      
     NNodeNudger::SaveBuffer("$TMP/NNodeNudger.npy"); 
     X4Transform3D::SaveBuffer("$TMP/X4Transform3D.npy"); 

     LOG(info) << " convertStructure END  " << m_sc->desc() ; 
}


GVolume* X4PhysicalVolume::convertTree_r(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const parent_pv, bool& recursive_select )
{
     GVolume* volume = convertNode(pv, parent, depth, parent_pv, recursive_select );
     m_ggeo->add(volume); // collect in nodelib

     const G4LogicalVolume* const lv = pv->GetLogicalVolume();
  
     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
     {
         const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
         convertTree_r(child_pv, volume, depth+1, pv, recursive_select );
     }

     return volume   ; 
}


unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
{
     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
     const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;

     const G4Material* const imat = lv->GetMaterial() ;
     const G4Material* const omat = lv_p ? lv_p->GetMaterial() : imat ;  // top omat -> imat 

     bool first_priority = true ;  
     const G4LogicalSurface* const isur = findSurface( pv  , pv_p , first_priority );
     const G4LogicalSurface* const osur = findSurface( pv_p, pv   , first_priority );  
     // doubtful of findSurface priority with double skin surfaces, see g4op-

     unsigned boundary = m_blib->addBoundary( 
                                                X4::BaseName(omat),  
                                                X4::BaseName(osur),                   
                                                X4::BaseName(isur),  
                                                X4::BaseName(imat)       
                                            );
     return boundary ; 
}


GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
{
     YOG::Nd* parent_nd = parent ? static_cast<YOG::Nd*>(parent->getParallelNode()) : NULL ;

     unsigned boundary = addBoundary( pv, pv_p );
     std::string boundaryName = m_blib->shortname(boundary); 
     int materialIdx = m_blib->getInnerMaterial(boundary); 

     LOG(trace) 
         << " boundary " << std::setw(4) << boundary 
         << " materialIdx " << std::setw(4) << materialIdx
         << " boundaryName " << boundaryName
         ;

     //materialIdx = m_ndCount ; // <-- checking effect of different material idx

     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
     const G4VSolid* const solid = lv->GetSolid();

     int lvIdx = m_lvidx[lv] ;   // from postorder IndexTraverse, to match GDML lvIdx : mesh identity uses lvIdx

     glm::mat4 xf_local = X4Transform3D::GetObjectTransform(pv);  
     const nmat4triple* ltriple = m_xform->make_triple( glm::value_ptr(xf_local) ) ; 

     const std::string& lvName = lv->GetName() ; 
     const std::string& pvName = pv->GetName() ; 
     const std::string& soName = solid->GetName() ; 

     int ndIdx0 = m_sc->get_num_nodes();
     int ndIdx = m_sc->add_node(
                                 lvIdx, 
                                 materialIdx,
                                 lvName,
                                 pvName,
                                 soName,
                                 ltriple,
                                 boundaryName,
                                 depth,
                                 true,      // selected: not yet used in YOG machinery  
                                 parent_nd
                               );

     assert( ndIdx == ndIdx0) ; 

     Nd* nd = m_sc->get_node(ndIdx) ; 

     if(ndIdx % 1000 == 0) 
     LOG(info) << "convertNode " 
               << " ndIdx "  << std::setw(5) << ndIdx 
               << " soIdx "  << std::setw(5) << nd->soIdx 
               << " lvIdx "  << std::setw(5) << lvIdx 
               << " materialIdx "  << std::setw(5) << materialIdx 
               << " soName " << soName
               ;

     assert( ndIdx == int(m_ndCount) ); 
     m_ndCount += 1 ; 

     const nmat4triple* gtriple = nxform<YOG::Nd>::make_global_transform(nd) ; 
     glm::mat4 xf_global = gtriple->t ;
     GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));

     Mh* mh = m_sc->get_mesh_for_node( ndIdx );  // node->mesh via soIdx (the local mesh index)


     if(mh->csgnode == NULL)
     {
         convertSolid( lvIdx, mh, nd, solid);
     }

     assert( mh->csgnode ); 

     //unsigned csgdepth = mh->csgnode->maxdepth();  
     //unsigned lvr_lvIdx = csgdepth ; // misuse lvr: selection  (gives a black render)
     unsigned lvr_lvIdx = lvIdx ; 

     bool selected = m_query->selected(pvName.c_str(), ndIdx0, depth, recursive_select, lvr_lvIdx );
    
     LOG(trace) << " lv_lvIdx " << lvr_lvIdx
                << " selected " << selected
               ; 

     const GMesh* mesh = mh->mesh ;   // hmm AssimpGGeo::convertMeshes does deduping/fixing before inclusion in GVolume(GNode) 

     const NCSG* csg = mh->csg ; 

     GParts* pts = GParts::make( csg, boundaryName.c_str(), m_verbosity  );  // see GScene::createVolume 
     pts->setBndLib(m_blib);

     //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ CAN THIS BE DONE AT MESH LEVEL ^^^^^^^^^^^^^^^
     // DOES EACH GVolume needs its own GParts ?
     // LOG(error) << " make pts " << pts->id() ; 

     // metadata needs to be transferred, like in GScene ?

     NSensor* sensor = NULL ; 

     GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local));

     GVolume* volume = new GVolume(ndIdx, gtransform, mesh, boundary, sensor );

     // sensor = m_sensor_list ? m_sensor_list->findSensorForNode( ndIndex ) : NULL ; 
     volume->setSensor( sensor );   
     volume->setBoundary( boundary ); 
     volume->setSelected( selected );
    
     // TODO: rejig ctor args, to avoid needing the setters for array setup

     volume->setLevelTransform(ltransform);
     volume->setParallelNode( nd ); 
     volume->setParts( pts ); 
     volume->setPVName( pvName.c_str() );
     volume->setLVName( lvName.c_str() );
     volume->setName( pvName.c_str() );   // historically (AssimpGGeo) this was set to lvName, but pvName makes more sense for node node

     m_ggeo->countMeshUsage(nd->soIdx, ndIdx );


     if(parent) 
     {
         parent->addChild(volume);
         volume->setParent(parent);
     } 

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

void X4PhysicalVolume::convertSolid( int lvIdx, Mh* mh, const Nd* nd, const G4VSolid* const solid)
{
     // convert G4VSolid into nnode tree and balance it if overheight 
     assert(mh->csgnode == NULL) ;

     nnode* raw = X4Solid::Convert(solid)  ; 

     if(m_g4codegen) 
     {
         raw->dump_g4code(); 
         //X4CSG::GenerateTest( solid, X4::X4GEN_DIR , lvIdx ) ; 
         X4CSG::GenerateTest( solid, m_g4codegendir , lvIdx ) ; 
     }

     nnode* balanced = NTreeProcess<nnode>::Process(raw, nd->soIdx, lvIdx); 
     mh->csgnode = balanced ; 

     std::vector<unsigned> skips = {27, 29}; // soIdx     (formerly 33 too)
     // they are skipped because Geant4 polygonization fails for them 

     bool is_skip = std::find( skips.begin(), skips.end(), nd->soIdx ) != skips.end()  ; 

     if( is_skip )
     {
          LOG(error) << " is_skip " 
                     << " soIdx " << nd->soIdx  
                     << " lvIdx " << lvIdx
                     ;  
         //mh->csgnode->dump();
     }

     mh->mesh = is_skip ? X4Mesh::Placeholder(solid, nd->soIdx) : X4Mesh::Convert(solid, nd->soIdx ) ; 
     mh->vtx = mh->mesh->m_x4src_vtx ; 
     mh->idx = mh->mesh->m_x4src_idx ; 

     const NSceneConfig* config = NULL ; 
     mh->csg = NCSG::Adopt( mh->csgnode, config, nd->soIdx, lvIdx );   // Adopt exports nnode tree to m_nodes buffer in NCSG instance

     assert( mh->csg ) ; 
     assert( mh->csg->isUsedGlobally() );

     X4SolidRec rec(solid, raw, balanced, mh->csg, nd->soIdx, lvIdx );  
     m_solidrec.push_back( rec ) ; 

     m_ggeo->add( mh->mesh ) ; 
} 



void X4PhysicalVolume::dumpSolidRec(const char* msg) const 
{
    LOG(error) << msg ; 
    std::ostream& out = std::cout ;
    solidRecTable( out ); 
}

void X4PhysicalVolume::writeSolidRec() const 
{
    //std::string path = BFile::FormPath( X4::X4GEN_DIR, "solids.txt" ) ; 
    std::string path = BFile::preparePath( m_g4codegendir, "solids.txt", true ) ; 
    LOG(error) << " writeSolidRec " 
               << " g4codegendir [" << m_g4codegendir << "]"
               << " path [" << path << "]" ;  
    std::ofstream out(path.c_str());
    solidRecTable( out ); 
}

void X4PhysicalVolume::solidRecTable( std::ostream& out ) const 
{
    unsigned num_solid = m_solidrec.size() ; 
    out << "written by X4PhysicalVolume::solidRecTable " << std::endl ; 
    out << "num_solid " << num_solid << std::endl ; 
    for(unsigned i=0 ; i < num_solid ; i++)
    {
        const X4SolidRec& rec = m_solidrec[i] ; 
        out << rec.desc() << std::endl ; 
    }
}



