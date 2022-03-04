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
    else
    {
        pv = MakePhysical(name) ; 
    }
#else
    pv = MakePhysical(name) ; 
#endif
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

G4VPhysicalVolume* X4VolumeMaker::MakePhysical(const char* name)
{
    G4LogicalVolume* lv = MakeLogical(name) ; 

    bool grid = strstr(name, "Grid") != nullptr ; 
    bool cube = strstr(name, "Cube") != nullptr ; 

    G4VPhysicalVolume* pv = nullptr ; 

    if(grid)
    {
        pv =   WrapLVGrid(lv, 1, 1, 1 ); 
    }
    else if(cube)
    {
        pv =   WrapLVTranslate(lv, 100., 100., 100. ); 
    }
    else
    {
        pv =   WrapLVTranslate(lv, 0., 0., 0. ); 
    }

    return pv ; 
}

/**

  000
  001
  010
  011
  100
  101
  110
  111  


**/

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
    LOG(LEVEL) << " (nx,ny,nz) (" << nx << "," << ny << " " << nz << ")" ; 

    int extent = std::max(std::max(nx, ny), nz); 
    G4double sc = 500. ; 
    G4double halfside = sc*extent*3. ; 

    G4VPhysicalVolume* world_pv = WorldBox(halfside); 
    G4LogicalVolume* world_lv = world_pv->GetLogicalVolume(); 

    for(int ix=-nx ; ix < nx+1 ; ix++)
    for(int iy=-ny ; iy < ny+1 ; iy++)
    for(int iz=-nz ; iz < nz+1 ; iz++)
    {   
        const char* iname = GridName("item", ix, iy, iz, "" );    
        G4ThreeVector tla( sc*double(ix), sc*double(iy), sc*double(iz) );  
        G4VPhysicalVolume* pv_n = new G4PVPlacement(0, tla, lv ,iname,world_lv,false,0);
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


