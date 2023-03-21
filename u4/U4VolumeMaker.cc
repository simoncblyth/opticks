#include <sstream>
#include <csignal>

#include "G4Material.hh"
#include "G4ThreeVector.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Box.hh"
#include "G4Orb.hh"

#include "SStr.hh"
#include "SOpticksResource.hh"
#include "SPath.hh"
#include "ssys.h"
#include "sstr.h"
#include "SPlace.h"

#include "SLOG.hh"

#include "U4.hh"
#include "U4Material.hh"
#include "U4Surface.h"
#include "U4SolidMaker.hh"
#include "U4VolumeMaker.hh"
#include "U4RotationMatrix.h"
#include "U4Volume.h"
#include "U4GDML.h"
#include "U4ThreeVector.h"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif

#ifdef WITH_PMTFASTSIM
#include "PMTFastSim.hh"
#endif



const plog::Severity U4VolumeMaker::LEVEL = SLOG::EnvLevel("U4VolumeMaker", "DEBUG"); 
const char* U4VolumeMaker::GEOM = ssys::getenvvar("GEOM", "BoxOfScintillator"); 
const char* U4VolumeMaker::METH = nullptr ; 

std::string U4VolumeMaker::Desc() // static
{
    std::stringstream ss ; 
    ss << "U4VolumeMaker::Desc" ; 
    ss << " GEOM " << ( GEOM ? GEOM : "-" ) ; 
    ss << " METH " << ( METH ? METH : "-" ) ; 
#ifdef WITH_PMTSIM
    ss << " WITH_PMTSIM " ; 
#else
    ss << " not-WITH_PMTSIM " ; 
#endif
    std::string s = ss.str(); 
    return s ; 
}

std::string U4VolumeMaker::Desc( const G4ThreeVector& tla )
{
    std::stringstream ss ; 
    ss << " tla (" 
       <<  " " << std::fixed << std::setw(10) << std::setprecision(3) << tla.x()
       <<  " " << std::fixed << std::setw(10) << std::setprecision(3) << tla.y()
       <<  " " << std::fixed << std::setw(10) << std::setprecision(3) << tla.z()
       <<  ") " 
       ;

    std::string s = ss.str(); 
    return s ; 
}





/**
U4VolumeMaker::PV
---------------------

Invokes several PV getter methods the first to provide a PV wins.

PVG_
    load geometry from a gdmlpath
PVP_
    PMTSim getter, requiring WITH_PMTSIM macro to be set meaning that PMTSim pkg was found by CMake
PVF_
    PMTFastSim getter, requiring WITH_PMTFASTSIM macro, meaning that PMTFastSim pkg found by CMake
PVS_
    Specials provided locally 
PVL_
    For names starting with List, multiple volumes are created and arranged eg into a grid. 
    The names to create are extracted using the name argumnent as a comma delimited list
PV1_
    Places single lv once or duplicates it several times depending on 
    strings found in the name such as Grid,Cube,Xoff,Yoff,Zoff

**/

const G4VPhysicalVolume* U4VolumeMaker::PV(){ return PV(GEOM); }
const G4VPhysicalVolume* U4VolumeMaker::PV(const char* name)
{
    LOG(LEVEL) << "[" << name ; 
    const G4VPhysicalVolume* pv = nullptr ; 
    if(pv == nullptr) pv = PVG_(name); 
    if(pv == nullptr) pv = PVP_(name); 
    if(pv == nullptr) pv = PVF_(name); 
    if(pv == nullptr) pv = PVS_(name); 
    if(pv == nullptr) pv = PVL_(name); 
    if(pv == nullptr) pv = PV1_(name); 
    LOG_IF(error, pv == nullptr) << "returning nullptr for name [" << name << "]" ; 
    LOG(LEVEL) << "]" << name ; 
    return pv ; 
}

/**
U4VolumeMaker::PVG_
---------------------

Attempts to load geometry from a gdmlpath using the SOpticksResource::GDMLPath 
resolution method.  For example for a GEOM name of hello the gdmlpath is obtained
from the envvar hello_GDMLPath with nullptr being returned when there is no such 
envvar. 

Additionally using SOpticksResource::GDMLSub resolution when hello_GDMLSub 
envvar exists the string obtained is used to select a PV from within the loaded 
tree of volumes.  For example with GEOM of J000 the below envvar is checked::

    export J000_GDMLSub=HamamatsuR12860sMask_virtual0x:0:1000

Note that while loading GDML and then selecting a PV from it is a useful capability 
it is generally only useful for a first look at an issue or to isolate an issue. 
As typically to understand what is going wrong with a geometry it is necessary 
to iterate making changes to the geometry. In order to do that it is necessary 
to take control of the geometry defining code for example in j/PMTSim. 

**/

const char* U4VolumeMaker::PVG_WriteNames     = "U4VolumeMaker_PVG_WriteNames" ; 
const char* U4VolumeMaker::PVG_WriteNames_Sub = "U4VolumeMaker_PVG_WriteNames_Sub" ; 

const G4VPhysicalVolume* U4VolumeMaker::PVG_(const char* name)
{
    METH = "PVG_" ; 
    const char* gdmlpath = SOpticksResource::GDMLPath(name) ;   
    const char* sub = SOpticksResource::GEOMSub(name);  

    bool exists = gdmlpath && SPath::Exists(gdmlpath) ; 

    const G4VPhysicalVolume* loaded = exists ? U4GDML::Read(gdmlpath) : nullptr ; 

    if(loaded && ssys::getenvbool(PVG_WriteNames))
        U4Volume::WriteNames( loaded, SPath::Resolve("$TMP", PVG_WriteNames, DIRPATH));  
    
    const G4VPhysicalVolume* pv = loaded ; 

    if( loaded && sub ) 
    { 
        const G4VPhysicalVolume* pv_sub = U4Volume::FindPVSub( loaded, sub ) ;  
        G4LogicalVolume* lv_sub = pv_sub->GetLogicalVolume(); 
        std::vector<G4LogicalVolume*> lvs = {lv_sub} ; 
        pv = Wrap( name, lvs );  

        if(ssys::getenvbool(PVG_WriteNames_Sub))
            U4Volume::WriteNames( pv, SPath::Resolve("$TMP", PVG_WriteNames_Sub, DIRPATH));  
    }

    LOG(LEVEL) 
          << " name " << name 
          << " gdmlpath " << gdmlpath 
          << " exists " << exists
          << " loaded " << loaded
          << " sub " << ( sub ? sub : "-" ) 
          << " pv " << pv 
          ;  

    return pv ; 
}


/**
U4VolumeMaker::PVP_ : Get PMTSim PV
------------------------------------ 

PMTSim::HasManagerPrefix 
    returns true for names starting with one of: hama, nnvt, hmsk, nmsk, lchi

PMTSim::GetLV PMTSim::GetPV
     these methods act as go betweens to the underlying managers with prefixes
     that identify the managers offset  


TODO: need to generalize the wrapping  

**/

const G4VPhysicalVolume* U4VolumeMaker::PVP_(const char* name)
{
    METH = "PVP_" ; 
    const G4VPhysicalVolume* pv = nullptr ; 
#ifdef WITH_PMTSIM
    const char* geomlist = SOpticksResource::GEOMList(name);   // consult envvar name_GEOMList 
    std::vector<std::string> names ; 
    if( geomlist == nullptr )
    {
        names.push_back(name); 
    } 
    else
    {
        sstr::Split(geomlist, ',', names ); 
    }
    int num_names = names.size(); 

    LOG(LEVEL) 
         << "[ WITH_PMTSIM"
         << " geomlist [" << ( geomlist ? geomlist : "-" ) << "] "
         << " name [" << name << "] "
         << " num_names " << num_names 
         ; 

    std::vector<G4LogicalVolume*> lvs ; ; 

    for(int i=0 ; i < num_names ; i++)
    {
        const char* n = names[i].c_str(); 
        bool has_manager_prefix = PMTSim::HasManagerPrefix(n) ;
        LOG(LEVEL) << "[ WITH_PMTSIM n [" << n << "] has_manager_prefix " << has_manager_prefix ; 
        assert( has_manager_prefix ); 

        G4LogicalVolume* lv = PMTSim::GetLV(n) ; 
        LOG_IF(fatal, lv == nullptr ) << "PMTSim::GetLV returned nullptr for n [" << n << "]" ; 
        assert( lv ); 

        lvs.push_back(lv); 
    }

    pv = Wrap( name, lvs ) ;          

    LOG(LEVEL) << "]" ; 
#else
    LOG(info) << " not-WITH_PMTSIM name [" << name << "]" ; 
#endif
    return pv ; 
}

/**
U4VolumeMaker::PVF_
---------------------

HMM: Coupling between U4 and PMTSim/PMTFastSim 
should be reduced to a minimum. 

Can it be reduced back to just to Geant4 geometry access ? 

Extending to JUNO specifics like junoPMTOpticalModel seems a step too far. 

Also, more comfortable with PMTSim/PMTFastSim dependency 
being restricted to test executables, keeping it out 
of the U4 lib. Could do this with header-only U4 impls.  

**/

const G4VPhysicalVolume* U4VolumeMaker::PVF_(const char* name)
{
    METH = "PVF_" ; 
    const G4VPhysicalVolume* pv = nullptr ; 
#ifdef WITH_PMTFASTSIM
    bool has_manager_prefix = PMTFastSim::HasManagerPrefix(name) ;
    LOG(LEVEL) << "[ WITH_PMTFASTSIM name [" << name << "] has_manager_prefix " << has_manager_prefix ; 
    if(has_manager_prefix) 
    {
        G4LogicalVolume* lv = PMTFastSim::GetLV(name) ; 
        LOG_IF(fatal, lv == nullptr ) << "PMTFastSim::GetLV returned nullptr for name [" << name << "]" ; 
        assert( lv ); 

        pv = Wrap( name, lv ) ;          
    }
    LOG(LEVEL) << "]" ; 
#else
    LOG(info) << " not-WITH_PMTFASTSIM name [" << name << "]" ; 
#endif
    return pv ; 
}


const G4VPhysicalVolume* U4VolumeMaker::PVS_(const char* name)
{
    METH = "PVS_" ; 
    const G4VPhysicalVolume* pv = nullptr ; 
    if(strcmp(name,"BoxOfScintillator" ) == 0)      pv = BoxOfScintillator(1000.);   
    if(strcmp(name,"RaindropRockAirWater" ) == 0)   pv = RaindropRockAirWater();   
    if(strcmp(name,"RaindropRockAirWaterSD" ) == 0) pv = RaindropRockAirWaterSD();   
    if(strcmp(name,"RaindropRockAirWaterSmall" ) == 0) pv = RaindropRockAirWater();   
    return pv ; 
}
const G4VPhysicalVolume* U4VolumeMaker::PVL_(const char* name)
{
    METH = "PVL_" ; 
    if(!SStr::StartsWith(name, "List")) return nullptr  ; 
    std::vector<G4LogicalVolume*> lvs ; 
    LV(lvs, name + strlen("List") ); 
    const G4VPhysicalVolume* pv = WrapLVGrid(lvs, 1, 1, 1 ); 
    return pv ; 
}
const G4VPhysicalVolume* U4VolumeMaker::PV1_(const char* name)
{
    METH = "PV1_" ; 
    G4LogicalVolume* lv = LV(name) ; 
    LOG_IF(error, lv == nullptr) << " failed to access lv for name " << name ; 
    if(lv == nullptr) return nullptr ; 

    const G4VPhysicalVolume* pv = nullptr ; 
    bool grid = strstr(name, "Grid") != nullptr ; 
    bool cube = strstr(name, "Cube") != nullptr ; 
    bool xoff = strstr(name, "Xoff") != nullptr ; 
    bool yoff = strstr(name, "Yoff") != nullptr ; 
    bool zoff = strstr(name, "Zoff") != nullptr ; 

    if(grid)      pv =   WrapLVGrid(     lv, 1, 1, 1 ); 
    else if(cube) pv =   WrapLVCube(     lv, 100., 100., 100. ); 
    else if(xoff) pv =   WrapLVOffset(   lv, 200.,   0.,   0. ); 
    else if(yoff) pv =   WrapLVOffset(   lv,   0., 200.,   0. ); 
    else if(zoff) pv =   WrapLVOffset(   lv,   0.,   0., 200. ); 
    else          pv =   WrapLVCube(     lv,   0.,   0.,   0. ); 
    return pv ; 
}


/**
U4VolumeMaker::LV
---------------------------

Note the generality: 

* U4SolidMaker::Make can create many different solid shapes based on the start of the name
* U4Material::FindMaterialName extracts the material name by looking for material names within *name*

HMM: maybe provide PMTSim LV this way too ? Based on some prefix ? 

**/

G4LogicalVolume*  U4VolumeMaker::LV(const char* name)
{
    const G4VSolid* solid_  = U4SolidMaker::Make(name); 
    LOG_IF(error, solid_==nullptr) << " failed to access solid for name " << name ; 
    if(solid_ == nullptr) return nullptr ; 

    G4VSolid* solid = const_cast<G4VSolid*>(solid_); 
    const char* matname = U4Material::FindMaterialName(name) ; 
    G4Material* material = U4Material::Get( matname ? matname : U4Material::VACUUM ); 
    G4LogicalVolume* lv = new G4LogicalVolume( solid, material, name, 0,0,0,true ); 
    return lv ; 
}

/**
U4VolumeMaker::LV vector interface : creates multiple LV using delimited names string 
--------------------------------------------------------------------------------------
**/

void U4VolumeMaker::LV(std::vector<G4LogicalVolume*>& lvs , const char* names_, char delim )
{
    std::vector<std::string> names ; 
    SStr::Split(names_ , delim, names ); 
    unsigned num_names = names.size(); 
    LOG(LEVEL) << " names_ " << names_ << " num_names " << num_names ;  
    assert( num_names > 1 ); 

    for(unsigned i=0 ; i < num_names ; i++)
    {
        const char* name = names[i].c_str(); 
        G4LogicalVolume* lv = LV(name) ; 
        assert(lv); 
        lvs.push_back(lv);   
    }
}


/**
U4VolumeMaker::Wrap
-----------------------

Consults envvar ${GEOM}_GEOMWrap for the wrap config, 
which when present must be one of : AroundCylinder, AroundSphere, AroundCircle

**/

const G4VPhysicalVolume* U4VolumeMaker::Wrap( const char* name, std::vector<G4LogicalVolume*>& items_lv )
{
    const char* wrap = SOpticksResource::GEOMWrap(name);  
    LOG(LEVEL) << "[ name " << name << " GEOMWrap " << ( wrap ? wrap : "-" ) ; 
    const G4VPhysicalVolume* pv = wrap == nullptr ? WrapRockWater( items_lv ) : WrapAroundItem( name, items_lv, wrap );  
    LOG(LEVEL) << "] name " << name << " wrap " << ( wrap ? wrap : "-" ) << " pv " << ( pv ? "YES" : "NO" ) ; 
    return pv ;  
}

/**
U4VolumeMaker::WrapRockWater
-------------------------------

    +---------------Rock------------+
    |                               |
    |                               |
    |     +--------Water------+     |
    |     |                   |     |
    |     |                   |     |
    |     |      +-Item-+     |     |
    |     |      |      |     |     |
    |     |      |      |     |     |
    |     |      +------+     |     |
    |     |                   |     |
    |     |                   |     |
    |     +-------------------+     |
    |                               |
    |                               |
    +-------------------------------+


**/

const G4VPhysicalVolume* U4VolumeMaker::WrapRockWater( std::vector<G4LogicalVolume*>& items_lv )
{
    assert( items_lv.size() >= 1 ); 
    G4LogicalVolume* item_lv = items_lv[items_lv.size()-1] ;  

    LOG(LEVEL) << "[ items_lv.size " << items_lv.size()   ; 

    double rock_halfside  = ssys::getenv_<double>(U4VolumeMaker_WrapRockWater_Rock_HALFSIDE , 1000.); 
    double water_halfside = ssys::getenv_<double>(U4VolumeMaker_WrapRockWater_Water_HALFSIDE,  900. ); 

    std::vector<double>* _boxscale = ssys::getenv_vec<double>(U4VolumeMaker_WrapRockWater_BOXSCALE, "1,1,1" ); 
    if(_boxscale) assert(_boxscale->size() == 3 ); 
    const double* boxscale = _boxscale ? _boxscale->data() : nullptr ;  

    LOG(LEVEL) << U4VolumeMaker_WrapRockWater_Rock_HALFSIDE << " " << rock_halfside ; 
    LOG(LEVEL) << U4VolumeMaker_WrapRockWater_Water_HALFSIDE << " " << water_halfside ; 

    G4LogicalVolume*  rock_lv  = Box_(rock_halfside,  "Rock",  nullptr, boxscale );
    G4LogicalVolume*  water_lv = Box_(water_halfside, "Water", nullptr, boxscale );
 
    //const char* flip_axes = "Z" ; 
    const char* flip_axes = nullptr ; 
    const G4VPhysicalVolume* item_pv  = Place(item_lv,  water_lv, flip_axes );  assert( item_pv ); 


    LOG(LEVEL) << std::endl << U4Volume::Traverse( item_pv ); 

    std::vector<std::string>* vbs1 = nullptr ; 
    vbs1 = ssys::getenv_vec<std::string>(U4VolumeMaker_WrapRockWater_BS1, "pyrex_vacuum_bs:hama_body_phys:hama_inner1_phys", ':' );

    G4LogicalBorderSurface* bs1 = nullptr ; 
    if(vbs1 && vbs1->size() == 3 )
    {
         const std::string& bs1n = (*vbs1)[0] ; 
         const std::string& pv1 = (*vbs1)[1] ; 
         const std::string& pv2 = (*vbs1)[2] ; 
         bs1 = U4Surface::MakePerfectDetectorBorderSurface(bs1n.c_str(), pv1.c_str(), pv2.c_str(), item_pv ); 

        LOG(error)
            << " attempt to add  PerfectDetectorBorderSurface between volumes "
            << " bs1n " << bs1n
            << " pv1 " << pv1 
            << " pv2 " << pv2 
            << " bs1 " << ( bs1 ? "YES" : "NO" )
            << " using config from " << U4VolumeMaker_WrapRockWater_BS1 
            ;
    }
    //if(bs1 == nullptr) std::raise(SIGINT); 


    const G4VPhysicalVolume* water_pv = Place(water_lv,  rock_lv);  assert( water_pv ); 
    const G4VPhysicalVolume* rock_pv  = Place(rock_lv,  nullptr );  

    G4LogicalBorderSurface* water_rock_bs = U4Surface::MakePerfectAbsorberBorderSurface("water_rock_bs", water_pv, rock_pv );  
    assert( water_rock_bs ); 

    LOG(LEVEL) << "]"  ; 
    return rock_pv ; 
}

/**
U4VolumeMaker::WrapAroundItem
-------------------------------

The *item_lv* is repeated many times using transforms from U4VolumeMaker::MakeTransforms
controlled by the *prefix* string which must be one of::

    AroundSphere
    AroundCylinder
    AroundCircle

which makes use of SPlace::AroundSphere SPlace::AroundCylinder SPlace::AroundCircle.
All those repeats have a "Water" box mother volume which is contained within "Rock". 

**/

NP* U4VolumeMaker::TRS = nullptr ; 

const G4VPhysicalVolume* U4VolumeMaker::WrapAroundItem( const char* name, std::vector<G4LogicalVolume*>& items_lv, const char* prefix )
{
    assert( items_lv.size() >= 1 ); 

    NP* trs = MakeTransforms(name, prefix) ; 
    TRS = trs ; 

    LOG(LEVEL) 
        << " items_lv.size " << items_lv.size() 
        << " prefix " << prefix 
        << " trs " << ( trs ? trs->sstr() : "-" )
        ;


    double rock_halfside   = ssys::getenv_<double>(U4VolumeMaker_WrapAroundItem_Rock_HALFSIDE,  20000.); 
    double water_halfside  = ssys::getenv_<double>(U4VolumeMaker_WrapAroundItem_Water_HALFSIDE, 19000.); 

    std::vector<double>* _rock_boxscale = ssys::getenv_vec<double>(U4VolumeMaker_WrapAroundItem_Rock_BOXSCALE, "1,1,1" ); 
    std::vector<double>* _water_boxscale = ssys::getenv_vec<double>(U4VolumeMaker_WrapAroundItem_Water_BOXSCALE, "1,1,1" ); 

    if(_rock_boxscale) assert(_rock_boxscale->size() == 3 ); 
    if(_water_boxscale) assert(_water_boxscale->size() == 3 ); 

    const double* rock_boxscale = _rock_boxscale ? _rock_boxscale->data() : nullptr ;  
    const double* water_boxscale = _water_boxscale ? _water_boxscale->data() : nullptr ;  

    G4LogicalVolume*  rock_lv  = Box_(rock_halfside,  "Rock" , nullptr, rock_boxscale );
    G4LogicalVolume*  water_lv = Box_(water_halfside, "Water", nullptr, water_boxscale );
 
    WrapAround(prefix, trs, items_lv, water_lv );  
    // item_lv placed inside water_lv once for each transform

    const G4VPhysicalVolume* water_pv = Place(water_lv,  rock_lv);  assert( water_pv ); 
    const G4VPhysicalVolume* rock_pv  = Place(rock_lv,  nullptr );  

    return rock_pv ; 
}

void U4VolumeMaker::SaveTransforms( const char* savedir ) // static
{
    NP* TRS = U4VolumeMaker::TRS ;
    if(TRS) TRS->save(savedir, "TRS.npy") ; 
}



/**
U4VolumeMaker::WrapVacuum
----------------------------

The LV provided is placed within a WorldBox of halfside extent and the world PV is returned. 

**/

const G4VPhysicalVolume* U4VolumeMaker::WrapVacuum( G4LogicalVolume* item_lv )
{
    double halfside = ssys::getenv_<double>(U4VolumeMaker_WrapVacuum_HALFSIDE, 1000.); 
    LOG(LEVEL) << U4VolumeMaker_WrapVacuum_HALFSIDE << " " << halfside ; 

    G4LogicalVolume*   vac_lv  = Box_(halfside, "Vacuum" );
 
    const G4VPhysicalVolume* item_pv = Place(item_lv, vac_lv);   assert( item_pv ); 
    const G4VPhysicalVolume* vac_pv  = Place(vac_lv, nullptr );

    return vac_pv ;      
}





/**
U4VolumeMaker::WrapLVGrid
---------------------------

Returns a physical volume with the argument lv placed multiple times 
in a grid specified by (nx,ny,nz) integers. (1,1,1) yields 3x3x3 grid.

The vector argument method places the lv in a grid within a box. 
Grid ranges::

    -nx:nx+1
    -ny:ny+1
    -nz:nz+1

Example (nx,ny,nz):

1,1,1 
     yields a grid with 3 elements on each side, for 3*3*3=27 
0,0,0
     yields a single element 

**/

const G4VPhysicalVolume* U4VolumeMaker::WrapLVGrid( G4LogicalVolume* lv, int nx, int ny, int nz  )
{
    std::vector<G4LogicalVolume*> lvs ; 
    lvs.push_back(lv); 
    return WrapLVGrid(lvs, nx, ny, nz ) ; 
}

const G4VPhysicalVolume* U4VolumeMaker::WrapLVGrid( std::vector<G4LogicalVolume*>& lvs, int nx, int ny, int nz  )
{
    unsigned num_lv = lvs.size(); 

    int extent = std::max(std::max(nx, ny), nz); 
    G4double sc = 500. ; 
    G4double halfside = sc*extent*3. ; 

    LOG(LEVEL) 
        << " num_lv " << num_lv 
        << " (nx,ny,nz) (" << nx << "," << ny << " " << nz << ")" 
        << " extent " << extent
        << " halfside " << halfside
        ;

    const G4VPhysicalVolume* world_pv = WorldBox(halfside); 
    G4LogicalVolume* world_lv = world_pv->GetLogicalVolume(); 

    unsigned count = 0 ; 
    for(int ix=-nx ; ix < nx+1 ; ix++)
    for(int iy=-ny ; iy < ny+1 ; iy++)
    for(int iz=-nz ; iz < nz+1 ; iz++)
    {   
        G4LogicalVolume* ulv = lvs[count%num_lv] ; 
        count += 1 ; 
        assert( ulv ); 

         const G4String& ulv_name_ = ulv->GetName();  
         std::string ulv_name = ulv_name_ ; 
         ulv_name += "_plc_" ; 

        const char* iname = GridName( ulv_name.c_str(), ix, iy, iz, "" );    
        G4ThreeVector tla( sc*double(ix), sc*double(iy), sc*double(iz) );  

        LOG(LEVEL) 
           << " G4PVPlacement "
           << " count " << std::setw(3) << count 
           << Desc(tla)
           <<  iname
           ;
        const G4VPhysicalVolume* pv_n = new G4PVPlacement(0, tla, ulv ,iname,world_lv,false,0);
        assert( pv_n );  
    }   
    return world_pv ; 
}




/**
U4VolumeMaker::WrapLVOffset
-----------------------------

This is used from PV1_ for Xoff Yoff Zoff names

1. use maximum of tx,ty,tz to define world box halfside 
2. places lv within world volume 
3. adds BoxMinusOrb at origin 

**/

const G4VPhysicalVolume* U4VolumeMaker::WrapLVOffset( G4LogicalVolume* lv, double tx, double ty, double tz )
{
    double halfside = 3.*std::max( std::max( tx, ty ), tz ); 
    assert( halfside > 0. ); 

    const G4VPhysicalVolume* world_pv = WorldBox(halfside); 
    G4LogicalVolume* world_lv = world_pv->GetLogicalVolume(); 

    AddPlacement(world_lv, lv, tx, ty, tz ); 

    bool bmo = true ; 
    if(bmo) AddPlacement( world_lv, "BoxMinusOrb", 0., 0., 0. ); 

    return world_pv ;  
}

/**
U4VolumeMaker::WrapLVCube
-------------------------------

This if used from PV1_ for "Cube" string.  
Places the lv at 8 positions at the corners of a cube. 

  ZYX
0 000
1 001
2 010          
3 011         
4 100        
5 101           
6 110
7 111

              110             111
                +-----------+
               /|          /| 
              / |         / | 
             /  |        /  |
            +-----------+   |
            |   +-------|---+ 011
            |  /        |  / 
            | /         | /
            |/          |/
            +-----------+
          000          001
            
   Z   Y
   | /   
   |/
   0-- X

**/

const G4VPhysicalVolume* U4VolumeMaker::WrapLVCube( G4LogicalVolume* lv, double tx, double ty, double tz )
{
    double halfside = 3.*std::max( std::max( tx, ty ), tz ); 

    const G4VPhysicalVolume* world_pv = WorldBox(halfside); 
    G4LogicalVolume* world_lv = world_pv->GetLogicalVolume(); 
    G4String name = lv->GetName(); 
    
    for(unsigned i=0 ; i < 8 ; i++)
    {
        bool px = ((i & 0x1) != 0) ; 
        bool py = ((i & 0x2) != 0) ; 
        bool pz = ((i & 0x4) != 0) ; 
        G4ThreeVector tla( px ? tx : -tx ,  py ? ty : -ty,  pz ? tz : -tz ); 
        const char* iname = GridName(name.c_str(), int(px), int(py), int(pz), "" );    
        const G4VPhysicalVolume* pv = new G4PVPlacement(0,tla,lv,iname,world_lv,false,0);
        assert( pv );  
    }
    return world_pv ;  
}




/**
U4VolumeMaker::AddPlacement : Translation places *lv* within *mother_lv*  
---------------------------------------------------------------------------

Used from U4VolumeMaker::WrapLVOffset

**/

const G4VPhysicalVolume* U4VolumeMaker::AddPlacement( G4LogicalVolume* mother_lv,  G4LogicalVolume* lv,  double tx, double ty, double tz )
{
    G4ThreeVector tla(tx,ty,tz); 
    const char* pv_name = SStr::Name( lv->GetName().c_str(), "_placement" ); 
    LOG(LEVEL) << Desc(tla) << " " << pv_name ;  
    const G4VPhysicalVolume* pv = new G4PVPlacement(0,tla,lv,pv_name,mother_lv,false,0);
    return pv ; 
} 
const G4VPhysicalVolume* U4VolumeMaker::AddPlacement( G4LogicalVolume* mother, const char* name,  double tx, double ty, double tz )
{
    G4LogicalVolume* lv = LV(name); 
    return AddPlacement( mother, lv, tx, ty, tz ); 
}

const char* U4VolumeMaker::GridName(const char* prefix, int ix, int iy, int iz, const char* suffix)
{
    std::stringstream ss ; 
    ss << prefix << ix << "_" << iy << "_" << iz << suffix ; 
    std::string s = ss.str();
    return strdup(s.c_str()); 
}

const char* U4VolumeMaker::PlaceName(const char* prefix, int ix, const char* suffix)
{
    std::stringstream ss ; 
    if(prefix) ss << prefix ; 
    ss << ix ;
    if(suffix) ss << suffix ; 
    std::string s = ss.str();
    return strdup(s.c_str()); 
}





/**
U4VolumeMaker::WorldBox
--------------------------

Used from U4VolumeMaker::WrapLVOffset U4VolumeMaker::WrapLVCube

**/

const G4VPhysicalVolume* U4VolumeMaker::WorldBox( double halfside, const char* mat )
{
    return Box(halfside, mat, "World", nullptr ); 
}
const G4VPhysicalVolume* U4VolumeMaker::BoxOfScintillator( double halfside )
{
    return BoxOfScintillator(halfside, "BoxOfScintillator", nullptr ); 
}
const G4VPhysicalVolume* U4VolumeMaker::BoxOfScintillator( double halfside, const char* prefix, G4LogicalVolume* mother_lv )
{
    return Box(halfside, U4Material::SCINTILLATOR, "BoxOfScintillator", mother_lv);
}
const G4VPhysicalVolume* U4VolumeMaker::Box(double halfside, const char* mat, const char* prefix, G4LogicalVolume* mother_lv )
{
    if(prefix == nullptr) prefix = mat ; 
    G4LogicalVolume* lv = Box_(halfside, mat, prefix); 
    return Place(lv, mother_lv); 
}


const G4VPhysicalVolume* U4VolumeMaker::Place( G4LogicalVolume* lv, G4LogicalVolume* mother_lv, const char* flip_axes )
{
    const char* lv_name = lv->GetName().c_str() ; 
    const char* pv_name = SStr::Name(lv_name, "_pv") ; 

    U4RotationMatrix* flip = flip_axes ? U4RotationMatrix::Flip(flip_axes) : nullptr ; 
    return new G4PVPlacement(flip,G4ThreeVector(), lv, pv_name, mother_lv, false, 0);
}

/**
U4VolumeMaker::RaindropRockAirWater
-------------------------------------

cf CSG/CSGMaker.cc CSGMaker::makeBoxedSphere


   +------------------------+ 
   | Rock                   |
   |    +-----------+       |
   |    | Air       |       |    
   |    |    . .    |       |    
   |    |   . Wa.   |       |    
   |    |    . .    |       |    
   |    |           |       |    
   |    +-----|-----+       |
   |                        |
   +----------|--rock_halfs-+ 
                   

Defaults::

    HALFSIDE: 100. 
    FACTOR: 1. 
   
    water_radius   :  halfside/2.         : 50. 
    air_halfside   :  halfside*factor     : 100.
    rock_halfside  :  1.1*halfside*factor : 110. 

An easy way to get some scattering and absorption to happen 
is to increase U4VolumeMaker_RaindropRockAirWater_FACTOR to 10. for example. 

**/

void U4VolumeMaker::RaindropRockAirWater_Configure( double& rock_halfside, double& air_halfside, double& water_radius )
{
    double halfside = ssys::getenv_<double>(U4VolumeMaker_RaindropRockAirWater_HALFSIDE, 100.); 
    double factor   = ssys::getenv_<double>(U4VolumeMaker_RaindropRockAirWater_FACTOR,   1.); 

    LOG(LEVEL) << U4VolumeMaker_RaindropRockAirWater_HALFSIDE << " " << halfside ; 
    LOG(LEVEL) << U4VolumeMaker_RaindropRockAirWater_FACTOR   << " " << factor ; 
 
    rock_halfside = 1.1*halfside*factor ; 
    air_halfside = halfside*factor ; 
    water_radius = halfside/2. ; 
}

const G4VPhysicalVolume* U4VolumeMaker::RaindropRockAirWater()
{
    double rock_halfside, air_halfside, water_radius ; 
    RaindropRockAirWater_Configure( rock_halfside, air_halfside, water_radius); 

    bool warn = false ; 
    G4Material* water_material  = G4Material::GetMaterial("Water", warn);   assert(water_material); 
    G4Material* air_material  = G4Material::GetMaterial("Air", warn);   assert(air_material); 
    G4Material* rock_material = G4Material::GetMaterial("Rock", warn);  assert(rock_material); 

    G4Orb* water_solid = new G4Orb("water_solid", water_radius ); 
    G4Box* air_solid = new G4Box("air_solid", air_halfside, air_halfside, air_halfside );
    G4Box* rock_solid = new G4Box("rock_solid", rock_halfside, rock_halfside, rock_halfside );

    G4LogicalVolume* water_lv = new G4LogicalVolume( water_solid, water_material, "water_lv"); 
    G4LogicalVolume* air_lv = new G4LogicalVolume( air_solid, air_material, "air_lv"); 
    G4LogicalVolume* rock_lv = new G4LogicalVolume( rock_solid, rock_material, "rock_lv" ); 

    const G4VPhysicalVolume* water_pv = new G4PVPlacement(0,G4ThreeVector(), water_lv ,"water_pv", air_lv,false,0);
    const G4VPhysicalVolume* air_pv = new G4PVPlacement(0,G4ThreeVector(),   air_lv ,  "air_pv",  rock_lv,false,0);
    const G4VPhysicalVolume* rock_pv = new G4PVPlacement(0,G4ThreeVector(),  rock_lv ,  "rock_pv", nullptr,false,0);

    assert( water_pv ); 
    assert( air_pv ); 
    assert( rock_pv ); 

    G4LogicalBorderSurface* air_rock_bs = U4Surface::MakePerfectAbsorberBorderSurface("air_rock_bs", air_pv, rock_pv );  
    assert( air_rock_bs ); 

    return rock_pv ; 
}

/**
U4VolumeMaker::RaindropRockAirWaterSD
--------------------------------------

Notice that so long as all the LV are created prior to creating the PV, 
which need the LV for placement and mother logical, there is no need to be 
careful with creation order of the volumes. 

This is suggestive of how to organize the API, instead of focussing on methods 
to create PV it is more flexible to have API that create LV that are then put 
together by the higher level methods that make less sense to generalize. 

**/

const G4VPhysicalVolume* U4VolumeMaker::RaindropRockAirWaterSD()
{
    double rock_halfside, air_halfside, water_radius ; 
    RaindropRockAirWater_Configure( rock_halfside, air_halfside, water_radius ); 

    G4LogicalVolume* rock_lv  = Box_(rock_halfside, "Rock" ); 
    G4LogicalVolume* air_lv   = Box_(air_halfside, "Air" ); 
    G4LogicalVolume* water_lv = Orb_(water_radius, "Water" ); 

    const G4VPhysicalVolume* rock_pv  = new G4PVPlacement(0,G4ThreeVector(), rock_lv ,  "rock_pv", nullptr,false,0);
    const G4VPhysicalVolume* air_pv   = new G4PVPlacement(0,G4ThreeVector(), air_lv  ,  "air_pv",  rock_lv,false,0);
    const G4VPhysicalVolume* water_pv = new G4PVPlacement(0,G4ThreeVector(), water_lv , "water_pv", air_lv,false,0);

    assert( rock_pv ); 
    assert( air_pv ); 
    assert( water_pv ); 

    G4LogicalBorderSurface* air_rock_bs = U4Surface::MakePerfectAbsorberBorderSurface("air_rock_bs", air_pv, rock_pv );  
    assert( air_rock_bs ); 

    //water_lv->SetSensitiveDetector(new U4SensitiveDetector("water_sd")); 
    // not needed, it is the below surface that is necessary 

    G4LogicalBorderSurface* air_water_bs = U4Surface::MakePerfectDetectorBorderSurface("air_water_bs", air_pv, water_pv );  
    assert( air_water_bs ); 

    return rock_pv ; 
}
G4LogicalVolume* U4VolumeMaker::Orb_( double radius, const char* mat, const char* prefix )
{
    if( prefix == nullptr ) prefix = mat ; 
    G4Material* material  = U4Material::Get(mat);   assert(material); 
    G4Orb* solid = new G4Orb( SStr::Name(prefix,"_solid"), radius ); 
    G4LogicalVolume* lv = new G4LogicalVolume( solid, material, SStr::Name(prefix,"_lv")); 
    return lv ; 
}
G4LogicalVolume* U4VolumeMaker::Box_( double halfside, const char* mat, const char* prefix, const double* scale )
{
    if( prefix == nullptr ) prefix = mat ; 
    G4Material* material  = U4Material::Get(mat);   assert(material); 

    double hx = scale ? scale[0]*halfside : halfside ; 
    double hy = scale ? scale[1]*halfside : halfside ; 
    double hz = scale ? scale[2]*halfside : halfside ; 

    LOG(LEVEL) << " hx " << hx << " hy " << hy << " hz " << hz << " halfside " << halfside ; 

    G4Box* solid = new G4Box( SStr::Name(prefix,"_solid"), hx, hy, hz );
    G4LogicalVolume* lv = new G4LogicalVolume( solid, material, SStr::Name(prefix,"_lv")); 
    return lv ; 
}


NP* U4VolumeMaker::MakeTransforms( const char* name, const char* prefix )
{
    const char* opts = "TR,tr,R,T,r,t" ; 
    NP* trs = nullptr ; 

    if(strcmp(prefix, "AroundSphere")==0)   
    {
        double radius = 17000. ; 
        double item_arclen = 600. ;   // 400. has lots of overlap, 1000. too spaced 
        unsigned num_ring = 8 ; 
        trs = SPlace::AroundSphere(  opts, radius, item_arclen, num_ring );
    }
    else if(strcmp(prefix, "AroundCylinder")==0) 
    {
        double radius = 17000. ; 
        double halfheight = radius ; 
        unsigned num_ring = 8 ; 
        unsigned num_in_ring = 16 ; 
        trs = SPlace::AroundCylinder(opts, radius, halfheight, num_ring, num_in_ring  );
    }
    else if(strcmp(prefix, "AroundCircle")==0) 
    {
        double radius = ssys::getenv_<double>(U4VolumeMaker_MakeTransforms_AroundCircle_radius, 1000.); 
        unsigned numInRing = ssys::getenv_<unsigned>(U4VolumeMaker_MakeTransforms_AroundCircle_numInRing, 4u );  
        double fracPhase = ssys::getenv_<double>(U4VolumeMaker_MakeTransforms_AroundCircle_fracPhase, 0.); 
        trs = SPlace::AroundCircle(opts, radius, numInRing, fracPhase  );
    }
    return trs ; 
}

/**
U4VolumeMaker::WrapAround
----------------------------

Place *lv* multiple times inside *mother_lv* using the *trs* 
tranforms to configure the placement rotations and translations. 

NB there is no check that the placements are within the mother_lv 
so it is up to the user to ensure that the mother volume is sufficiently 
large to accommodate the lv with all the transforms applied to it. 

**/

void U4VolumeMaker::WrapAround( const char* prefix, const NP* trs, std::vector<G4LogicalVolume*>& lvs, G4LogicalVolume* mother_lv )
{
    unsigned num_place = trs->shape[0] ; 
    unsigned place_tr = trs->shape[1] ;   
    unsigned place_values = place_tr*4*4 ; 
    unsigned num_lv = lvs.size(); 


    assert( trs->has_shape(num_place,place_tr,4,4) );  
    assert( place_tr == 6 );  // expected number of different options from "TR,tr,R,T,r,t"
    enum { _TR, _tr, _R, _T, _r, _t } ;  // order must match opts "TR,tr,R,T,r,t"

    const double* tt = trs->cvalues<double>(); 

    for(unsigned i=0 ; i < num_place ; i++)
    {
        const double* T = tt + place_values*i + _T*16 ;  
        const double* R = tt + place_values*i + _R*16 ;  

        // TODO: get these from a single matrix, not 6 

        G4ThreeVector tla( T[12], T[13], T[14] ); 

        LOG(LEVEL) << " i " << std::setw(7) << " tla " << U4ThreeVector::Desc(tla) ; 

        bool transpose = true ; 
        U4RotationMatrix* rot = new U4RotationMatrix( R, transpose );  // ISA G4RotationMatrix

        LOG(LEVEL) << " i " << std::setw(7) << " rot " << U4RotationMatrix::Desc(rot) ; 

        const char* iname = PlaceName(prefix, i, nullptr); 

        G4bool pMany_unused = false ; 
        G4int  pCopyNo = (i+1)*10 ; 
        G4LogicalVolume* lv = lvs[i%num_lv] ; 

        const G4VPhysicalVolume* pv_n = new G4PVPlacement(rot, tla, lv, iname, mother_lv, pMany_unused, pCopyNo ); // 1st ctor
        assert( pv_n );  
    }
}

