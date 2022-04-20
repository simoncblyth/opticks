#include <sstream>

#include "G4Material.hh"
#include "G4ThreeVector.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Box.hh"

#include "SStr.hh"
#include "PLOG.hh"

#include "X4Material.hh"
#include "X4SolidMaker.hh"
#include "X4VolumeMaker.hh"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif

const plog::Severity X4VolumeMaker::LEVEL = PLOG::EnvLevel("X4VolumeMaker", "DEBUG"); 

G4VPhysicalVolume* X4VolumeMaker::Make(const char* name)
{
    G4VPhysicalVolume* pv = nullptr ; 
    
#ifdef WITH_PMTSIM
    if(PMTSim::HasManagerPrefix(name))
    {
        G4LogicalVolume* lv = PMTSim::GetLV(name); 
        double halfside = 5000. ;  
        pv = Wrap(lv, halfside) ;          
    }
#endif
    if( pv == nullptr) pv = MakePhysical(name) ; 
    return pv ; 
}

G4LogicalVolume*  X4VolumeMaker::MakeLogical(const char* name)
{
    const G4VSolid* solid_  = X4SolidMaker::Make(name); 
    G4VSolid* solid = const_cast<G4VSolid*>(solid_); 

    G4Material* material = X4Material::Get("Vacuum"); 
    G4LogicalVolume* lv = new G4LogicalVolume( solid, material, name, 0,0,0,true ); 
    return lv ; 
}

void X4VolumeMaker::MakeLogical(std::vector<G4LogicalVolume*>& lvs , const char* names_ )
{
    std::vector<std::string> names ; 
    SStr::Split(names_ , ',', names ); 
    unsigned num_names = names.size(); 
    LOG(LEVEL) << " names_ " << names_ << " num_names " << num_names ;  
    assert( num_names > 1 ); 

    for(unsigned i=0 ; i < num_names ; i++)
    {
        const char* name = names[i].c_str(); 
        G4LogicalVolume* lv = MakeLogical(name) ; 
        assert(lv); 
        lvs.push_back(lv);   
    }
}

G4VPhysicalVolume* X4VolumeMaker::MakePhysical(const char* name)
{
    bool list = strstr(name, "List") != nullptr ; 
    G4VPhysicalVolume* pv = list ? MakePhysicalList_(name) : MakePhysicalOne_(name)  ; 
    return pv ; 
}


G4VPhysicalVolume* X4VolumeMaker::MakePhysicalList_(const char* name)
{
    assert( SStr::StartsWith(name, "List") ); 
    std::vector<G4LogicalVolume*> lvs ; 
    MakeLogical(lvs, name + strlen("List") ); 
    G4VPhysicalVolume* pv = WrapLVGrid(lvs, 1, 1, 1 ); 
    return pv ; 
}

G4VPhysicalVolume* X4VolumeMaker::MakePhysicalOne_(const char* name)
{
    G4VPhysicalVolume* pv = nullptr ; 

    bool grid = strstr(name, "Grid") != nullptr ; 
    bool cube = strstr(name, "Cube") != nullptr ; 
    bool xoff = strstr(name, "Xoff") != nullptr ; 
    bool yoff = strstr(name, "Yoff") != nullptr ; 
    bool zoff = strstr(name, "Zoff") != nullptr ; 

    G4LogicalVolume* lv = MakeLogical(name) ; 

    if(grid)      pv =   WrapLVGrid(lv, 1, 1, 1 ); 
    else if(cube) pv =   WrapLVTranslate(lv, 100., 100., 100. ); 
    else if(xoff) pv =   WrapLVOffset(lv, 200.,   0.,   0. ); 
    else if(yoff) pv =   WrapLVOffset(lv,   0., 200.,   0. ); 
    else if(zoff) pv =   WrapLVOffset(lv,   0.,   0., 200. ); 
    else          pv =   WrapLVTranslate(lv, 0., 0., 0. ); 

    return pv ; 
}



void X4VolumeMaker::AddPlacement( G4LogicalVolume* mother,  G4LogicalVolume* lv,  double tx, double ty, double tz )
{
    G4ThreeVector tla(tx,ty,tz); 
    std::string name = lv->GetName(); 
    name += "_placement" ; 
    LOG(LEVEL) << Desc(tla) << " " << name ;  
    G4VPhysicalVolume* pv = new G4PVPlacement(0,tla,lv,name,mother,false,0);
    assert(pv); 
} 


void X4VolumeMaker::AddPlacement( G4LogicalVolume* mother, const char* name,  double tx, double ty, double tz )
{
    G4LogicalVolume* lv = MakeLogical(name); 
    AddPlacement( mother, lv, tx, ty, tz ); 
}

G4VPhysicalVolume* X4VolumeMaker::WrapLVOffset( G4LogicalVolume* lv, double tx, double ty, double tz )
{
    G4String name = lv->GetName(); 
    name += "_Offset" ; 

    double halfside = 3.*std::max( std::max( tx, ty ), tz ); 
    G4VPhysicalVolume* world_pv = WorldBox(halfside); 
    G4LogicalVolume* world_lv = world_pv->GetLogicalVolume(); 

    AddPlacement(world_lv, lv, tx, ty, tz ); 

    bool bmo = true ; 
    if(bmo) AddPlacement( world_lv, "BoxMinusOrb", 0., 0., 0. ); 

    return world_pv ;  
}

G4VPhysicalVolume* X4VolumeMaker::WrapLVTranslate( G4LogicalVolume* lv, double tx, double ty, double tz )
{
    double halfside = 3.*std::max( std::max( tx, ty ), tz ); 

    G4VPhysicalVolume* world_pv = WorldBox(halfside); 
    G4LogicalVolume* world_lv = world_pv->GetLogicalVolume(); 
    G4String name = lv->GetName(); 
    
    for(unsigned i=0 ; i < 8 ; i++)
    {
        bool px = ((i & 0x1) != 0) ; 
        bool py = ((i & 0x2) != 0) ; 
        bool pz = ((i & 0x4) != 0) ; 
        G4ThreeVector tla( px ? tx : -tx ,  py ? ty : -ty,  pz ? tz : -tz ); 
        const char* iname = GridName(name.c_str(), int(px), int(py), int(pz), "" );    
        G4VPhysicalVolume* pv = new G4PVPlacement(0,tla,lv,iname,world_lv,false,0);
        assert( pv );  
    }
    return world_pv ;  
}

G4VPhysicalVolume* X4VolumeMaker::WorldBox( double halfside )
{
    G4Box* solid = new G4Box("World_solid", halfside, halfside, halfside );  
    G4Material* vacuum = X4Material::Get("Vacuum"); 
    G4LogicalVolume* lv = new G4LogicalVolume(solid,vacuum,"World_lv",0,0,0); 
    G4VPhysicalVolume* pv = new G4PVPlacement(0,G4ThreeVector(), lv ,"World_pv",0,false,0);
    return pv ; 
}


/**
X4VolumeMaker::WrapLVGrid
---------------------------

Returns a physical volume with the argument lv placed multiple times 
in a grid specified by (nx,ny,nz) integers.

**/

G4VPhysicalVolume* X4VolumeMaker::WrapLVGrid( G4LogicalVolume* lv, int nx, int ny, int nz  )
{
    std::vector<G4LogicalVolume*> lvs ; 
    lvs.push_back(lv); 
    return WrapLVGrid(lvs, nx, ny, nz ) ; 
}


std::string X4VolumeMaker::Desc( const G4ThreeVector& tla )
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


G4VPhysicalVolume* X4VolumeMaker::WrapLVGrid( std::vector<G4LogicalVolume*>& lvs, int nx, int ny, int nz  )
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

    G4VPhysicalVolume* world_pv = WorldBox(halfside); 
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


        G4VPhysicalVolume* pv_n = new G4PVPlacement(0, tla, ulv ,iname,world_lv,false,0);
        assert( pv_n );  
    }   
    return world_pv ; 
}


const char* X4VolumeMaker::GridName(const char* prefix, int ix, int iy, int iz, const char* suffix)
{
    std::stringstream ss ; 
    ss << prefix << ix << "_" << iy << "_" << iz << suffix ; 
    std::string s = ss.str();
    return strdup(s.c_str()); 
}

G4VPhysicalVolume* X4VolumeMaker::Wrap( G4LogicalVolume* lv, double halfside  )
{
    G4VPhysicalVolume* world_pv = WorldBox(halfside); 
    G4LogicalVolume* world_lv = world_pv->GetLogicalVolume(); 

    G4String name = lv->GetName(); 
    name += "_pv" ; 

    G4ThreeVector tla(0.,0.,0.); 
    G4VPhysicalVolume* pv_item = new G4PVPlacement(0, tla, lv ,name,world_lv,false,0);
    assert( pv_item );
  

    return world_pv ; 
}


