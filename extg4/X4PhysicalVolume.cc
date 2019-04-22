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
#include "BOpticksKey.hh"

class NSensor ; 

#include "NXform.hpp"  // header with the implementation
template struct nxform<X4Nd> ; 

#include "NGLMExt.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NNodeNudger.hpp"
#include "NTreeProcess.hpp"

#include "GMesh.hh"
#include "GVolume.hh"
#include "GParts.hh"
#include "GGeo.hh"
#include "GGeoSensor.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GSkinSurface.hh"
#include "GBndLib.hh"

#include "Opticks.hh"
#include "OpticksQuery.hh"


const plog::Severity X4PhysicalVolume::LEVEL = debug ;



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
    m_xform(new nxform<X4Nd>(0,false)),
    m_verbosity(m_ok->getVerbosity()),
    m_node_count(0),
    m_selected_node_count(0)
{
    const char* msg = "GGeo ctor argument of X4PhysicalVolume must have mlib, slib and blib already " ; 

    // trying to Opticks::configure earlier, from Opticks::init trips these asserts
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
    convertSensors();  // before closeSurfaces as may add some SensorSurfaces
    closeSurfaces();
    convertSolids();
    convertStructure();
    convertCheck();
}


/**
X4PhysicalVolume::convertSensors
---------------------------------

Predecessor in old route is AssimpGGeo::convertSensors

**/

void X4PhysicalVolume::convertSensors()
{
    LOG(debug) << "[" ; 

    convertSensors_r(m_top, 0); 

    unsigned num_clv = m_ggeo->getNumCathodeLV();
    unsigned num_bds = m_ggeo->getNumBorderSurfaces() ; 
    unsigned num_sks0 = m_ggeo->getNumSkinSurfaces() ; 

    GGeoSensor::AddSensorSurfaces(m_ggeo) ;

    unsigned num_sks1 = m_ggeo->getNumSkinSurfaces() ; 
    assert( num_bds == m_ggeo->getNumBorderSurfaces()  ); 

    LOG(debug) << "]" ; 

    LOG(info) 
         << " m_lvsdname " << m_lvsdname 
         << " num_clv " << num_clv 
         << " num_bds " << num_bds
         << " num_sks0 " << num_sks0
         << " num_sks1 " << num_sks1
         ; 


}


/**
X4PhysicalVolume::convertSensors_r
-----------------------------------

Sensors are identified by two approaches:

1. logical volume having an associated sensitive detector G4VSensitiveDetector
2. name of logical volume matching one of a comma delimited list 
   of strings provided by the "LV sensitive detector name" option
   eg  "--lvsdname Cathode,cathode,Sensor,SD" 

The second approach is useful as a workaround when operating 
with a GDML loaded geometry, as GDML does not yet(?) persist 
the SD LV association.

Names of sensitive LV are inserted into a set datastructure in GGeo. 

**/

void X4PhysicalVolume::convertSensors_r(const G4VPhysicalVolume* const pv, int depth)
{
    const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    const char* lvname = lv->GetName().c_str(); 
    G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ; 

    bool is_lvsdname = m_lvsdname && BStr::Contains(lvname, m_lvsdname, ',' ) ;
    bool is_sd = sd != NULL ; 

    const std::string sdn = sd ? sd->GetName() : "SD?" ;   // perhaps GetFullPathName() 

    if( is_lvsdname || is_sd )
    {
        std::string name = BFile::Name(lvname); 
        std::string nameref = SGDML::GenerateName( name.c_str() , lv , true );   
        LOG(info) 
            << " is_lvsdname " << is_lvsdname
            << " is_sd " << is_sd
            << " name " << name 
            << " nameref " << nameref 
            ;

        m_ggeo->addLVSD(nameref.c_str(), sdn.c_str()) ;
    }  

    for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    {
        const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
        convertSensors_r(child_pv, depth+1 );
    }
}


void X4PhysicalVolume::convertMaterials()
{
    LOG(verbose) << "[" ;

    size_t num_materials0 = m_mlib->getNumMaterials() ;
    assert( num_materials0 == 0 );

    X4MaterialTable::Convert(m_mlib);

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
          ; 
}

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

    LOG(verbose) << "]" ;

    LOG(info) 
           << " num_lbs " << num_lbs
           << " num_sks " << num_sks
           ;

}

void X4PhysicalVolume::closeSurfaces()
{
    m_slib->close();  // may change order if prefs dictate
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

The entire tree volume tree is recursed, but only the 
first occurence of each LV solid gets converted 
(because they are all the same).
Done this way to have consistent lvIdx soIdx indexing with GDML ?


**/

void X4PhysicalVolume::convertSolids()
{
    LOG(info) << "[" ; 

    const G4VPhysicalVolume* pv = m_top ; 
    int depth = 0 ;
    convertSolids_r(pv, depth);

    if(m_verbosity > 5) dumpLV();
    LOG(debug) << "]" ; 

    dumpTorusLV();
    LOG(info) << "]" ;

}

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

        GMesh* mesh = convertSolid( lvIdx, soIdx, solid, lvname ) ;  
        mesh->setIndex( lvIdx ) ;   

        const nnode* root = mesh->getRoot(); 
        assert( root ); 

        if( root->has_torus() )
        {
            LOG(fatal) << " has_torus lvIdx " << lvIdx << " " << lvname ;  
            m_lv_with_torus.push_back( lvIdx ); 
            m_lvname_with_torus.push_back( lvname ); 
        }

        m_ggeo->add( mesh ) ; 
    }  
}

GMesh* X4PhysicalVolume::convertSolid( int lvIdx, int soIdx, const G4VSolid* const solid, const std::string& lvname) const 
{
     assert( lvIdx == soIdx );  
     LOG(info) << " [ " << lvIdx << " " << lvname ; 
 
     nnode* raw = X4Solid::Convert(solid)  ; 

     if(m_g4codegen) 
     {
         const char* gdmlpath = X4CSG::GenerateTestPath( m_g4codegendir, lvIdx, ".gdml" ) ;  
         bool refs = false ;  
         X4GDMLParser::Write( solid, gdmlpath, refs ); 

         LOG(info) 
             << "[--g4codegen]"
             << " lvIdx " << lvIdx
             << " soIdx " << soIdx
             << " lvname " << lvname 
             ;
         raw->dump_g4code();  // just for debug 
         X4CSG::GenerateTest( solid, m_g4codegendir , lvIdx ) ; 

     }

     nnode* root = NTreeProcess<nnode>::Process(raw, soIdx, lvIdx);  // balances deep trees
     root->other = raw ; 


     const NSceneConfig* config = NULL ; 
     NCSG* csg = NCSG::Adopt( root, config, soIdx, lvIdx );   // Adopt exports nnode tree to m_nodes buffer in NCSG instance
     assert( csg ) ; 
     assert( csg->isUsedGlobally() );

     const std::string& soname = solid->GetName() ; 
     csg->set_soname( soname.c_str() ) ; 
     csg->set_lvname( lvname.c_str() ) ; 

     bool is_x4polyskip = m_ok->isX4PolySkip(lvIdx);   // --x4polyskip 211,232
     if( is_x4polyskip ) LOG(fatal) << " is_x4polyskip " << " soIdx " << soIdx  << " lvIdx " << lvIdx ;  

     GMesh* mesh =  is_x4polyskip ? X4Mesh::Placeholder(solid ) : X4Mesh::Convert(solid ) ; 
     mesh->setCSG( csg ); 

     LOG(info) << " ] " << lvIdx ; 
     return mesh ; 
}

/**
convertSolid
-------------

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




void X4PhysicalVolume::dumpLV() const 
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
           << " idx " << std::setw(5) << m_lvidx.at(lv)  
           << " lv "  << lv->GetName()
           << std::endl ;  
   }
}

void X4PhysicalVolume::dumpTorusLV() const 
{
    assert( m_lv_with_torus.size() == m_lvname_with_torus.size() ); 
    unsigned num_afflicted = m_lv_with_torus.size() ;  

    LOG(info) << " num_afflicted " << num_afflicted ; 
    std::cout << " lvIdx ( " ; 
    for(unsigned i=0 ; i < num_afflicted ; i++) std::cout << m_lv_with_torus[i] << " " ; 
    std::cout << " ) " << std::endl ;  

    for(unsigned i=0 ; i < num_afflicted ; i++) 
    {
        std::cout 
            << m_lv_with_torus[i] << " "  
            << m_lvname_with_torus[i] 
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

Note that its the YOG model that is updated, that gets
converted to glTF later.  This is done to help keeping 
this code independant of the actual glTF implementation 
used.

* NB this is very similar to AssimpGGeo::convertStructure, GScene::createVolumeTree

**/


void X4PhysicalVolume::convertStructure()
{
    LOG(info) << "[" ; 
    assert(m_top) ;

    m_ggeo->dumpCathodeLV("dumpCathodeLV"); 
    m_ggeo->dumpSkinSurface("dumpSkinSurface"); 


    const G4VPhysicalVolume* pv = m_top ; 
    GVolume* parent = NULL ; 
    const G4VPhysicalVolume* parent_pv = NULL ; 
    int depth = 0 ;

    bool recursive_select = false ;
    m_root = convertStructure_r(pv, parent, depth, parent_pv, recursive_select );

    NTreeProcess<nnode>::SaveBuffer("$TMP/NTreeProcess.npy");      
    NNodeNudger::SaveBuffer("$TMP/NNodeNudger.npy"); 
    X4Transform3D::SaveBuffer("$TMP/X4Transform3D.npy"); 

    LOG(info) << "]" ;
}


GVolume* X4PhysicalVolume::convertStructure_r(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const parent_pv, bool& recursive_select )
{
     GVolume* volume = convertNode(pv, parent, depth, parent_pv, recursive_select );
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

See notes/issues/ab-blib.rst

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
    if( g_sslv == NULL && g_sslv_p == NULL  )
    {
        const char* osur = X4::BaseName( osur_ ); 
        const char* isur = X4::BaseName( isur_ ); 
        boundary = m_blib->addBoundary( omat, osur, isur, imat ); 
    }
    else if( g_sslv && !g_sslv_p )
    {
        const char* osur = g_sslv->getName(); 
        const char* isur = osur ; 
        boundary = m_blib->addBoundary( omat, osur, isur, imat ); 
    }
    else if( g_sslv_p && !g_sslv )
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


GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
{

     X4Nd* parent_nd = parent ? static_cast<X4Nd*>(parent->getParallelNode()) : NULL ;

     unsigned boundary = addBoundary( pv, pv_p );
     std::string boundaryName = m_blib->shortname(boundary); 
     int materialIdx = m_blib->getInnerMaterial(boundary); 

     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
     const std::string& lvName = lv->GetName() ; 
     const std::string& pvName = pv->GetName() ; 

     int lvIdx = m_lvidx[lv] ;   // from postorder traverse in convertSolids to match GDML lvIdx : mesh identity uses lvIdx

     LOG(verbose) 
         << " boundary " << std::setw(4) << boundary 
         << " materialIdx " << std::setw(4) << materialIdx
         << " boundaryName " << boundaryName
         << " lvIdx " << lvIdx
         ;

     const GMesh* mesh = m_ggeo->getMesh(lvIdx); 
     const NCSG* csg = mesh->getCSG();  
     GParts* pts = GParts::make( csg, boundaryName.c_str(), m_verbosity  );  // see GScene::createVolume 
     pts->setBndLib(m_blib);



     //  boundary name is a node level thing, not mesh level : so forced to do GParts::make at node level
     //  TODO: see if GParts stuff not needing the boundary could be factored out 

     glm::mat4 xf_local = X4Transform3D::GetObjectTransform(pv);  

     const nmat4triple* ltriple = m_xform->make_triple( glm::value_ptr(xf_local) ) ; 
     GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local));

     X4Nd* nd = new X4Nd { parent_nd, ltriple } ;        

     const nmat4triple* gtriple = nxform<X4Nd>::make_global_transform(nd) ; 
     glm::mat4 xf_global = gtriple->t ;
     GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));

     NSensor* sensor = NULL ; 
     unsigned ndIdx = m_node_count ;  

 
     pts->setVolumeIndex( ndIdx );  

     GVolume* volume = new GVolume(ndIdx, gtransform, mesh, boundary, sensor );
     m_node_count += 1 ; 

     unsigned lvr_lvIdx = lvIdx ; 
     bool selected = m_query->selected(pvName.c_str(), ndIdx, depth, recursive_select, lvr_lvIdx );
     if(selected) m_selected_node_count += 1 ;  

     LOG(verbose) << " lv_lvIdx " << lvr_lvIdx
                << " selected " << selected
               ; 
 
     volume->setSensor( sensor );   
     volume->setBoundary( boundary ); 
     volume->setSelected( selected );
     // TODO: rejig GVolume ctor args, to avoid the apparent duplication of setters for array setup

     volume->setLevelTransform(ltransform);

     volume->setLocalTransform(ltriple);
     volume->setGlobalTransform(gtriple);
 
     volume->setParallelNode( nd ); 
     volume->setParts( pts ); 
     volume->setPVName( pvName.c_str() );
     volume->setLVName( lvName.c_str() );
     volume->setName( pvName.c_str() );   // historically (AssimpGGeo) this was set to lvName, but pvName makes more sense for node node

     m_ggeo->countMeshUsage(lvIdx, ndIdx );


     if(parent) 
     {
         parent->addChild(volume);
         volume->setParent(parent);
     } 

     return volume ; 
}



