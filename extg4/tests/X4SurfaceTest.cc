#include "G4Box.hh"
#include "G4Orb.hh"
#include "G4Element.hh"
#include "G4Material.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4ThreeVector.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4SystemOfUnits.hh"

#include "GGeo.hh"
#include "X4PhysicalVolume.hh"

#include "OPTICKS_LOG.hh"

struct X4SurfaceTest
{
    static G4Material* MakeWater(); 
    static G4Material* MakeGlass(); 
    static G4OpticalSurface* MakeOpticalSurface(const char* name); 

    static std::string MakeName(const char* prefix, int ix, int iy, int iz); 

    static constexpr int nx = 5 ; 
    static constexpr int ny = 5 ; 
    static constexpr int nz = 5 ; 
    static unsigned Index(int ix, int iy, int iz);   
    static unsigned Count();   

    unsigned           num_module ; 
    G4Material*        water ; 
    G4Material*        glass ; 
    G4Box*             so_world ; 
    G4LogicalVolume*   lv_world ; 
    G4VPhysicalVolume* pv_world ; 
    G4Orb*             so_module ;  
    G4LogicalVolume**  lv_module ; 
    G4PVPlacement**    pv_module ; 
    G4LogicalSkinSurface**    ss_module ; 
    G4LogicalBorderSurface**  bs_module ; 

    X4SurfaceTest(); 
    void init(); 
}; 

unsigned X4SurfaceTest::Count()
{
    unsigned count = 0 ; 
    for(int ix=-nx ; ix <= nx ; ix++) 
    for(int iy=-ny ; iy <= ny ; iy++) 
    for(int iz=-nz ; iz <= nz ; iz++)
    {
        count += 1 ; 
    }
    return count ; 
}

unsigned X4SurfaceTest::Index(int ix, int iy, int iz)
{
    unsigned ii = ix+nx ; 
    unsigned jj = iy+ny ; 
    unsigned kk = iz+nz ; 

    //unsigned ni = 2*nx + 1 ; 
    unsigned nj = 2*ny + 1 ; 
    unsigned nk = 2*nz + 1 ; 

    return ii*nj*nk + jj*nk + kk ;
}


X4SurfaceTest::X4SurfaceTest()
    :
    num_module(Count()),
    water(MakeWater()),
    glass(MakeGlass()),
    so_world(new G4Box("World",1000.*1000.,1000.*1000.,1000.*1000.)),
    lv_world(new G4LogicalVolume(so_world,water,"lv_world",0,0,0)),
    pv_world(new G4PVPlacement(0,G4ThreeVector(),lv_world ,"pv_world",0,false,0)),
    so_module(new G4Orb("so_module", 500. )),
    lv_module(new G4LogicalVolume*[num_module]),
    pv_module(new G4PVPlacement*[num_module]),
    ss_module(new G4LogicalSkinSurface*[num_module]),
    bs_module(new G4LogicalBorderSurface*[num_module])
{
    init(); 
}


G4Material* X4SurfaceTest::MakeWater()
{
    G4double a, z, density;
    G4int nelements;
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);
    G4Element* H = new G4Element("Hydrogen", "H", z=1 , a=1.01*CLHEP::g/CLHEP::mole);
    G4Material* mat = new G4Material("Water", density= 1.0*CLHEP::g/CLHEP::cm3, nelements=2);
    mat->AddElement(H, 2);
    mat->AddElement(O, 1);
    return mat ;
}

G4Material* X4SurfaceTest::MakeGlass()
{
    G4double a, z, density;
    G4int nelements;
    G4Element* H = new G4Element("H", "H", z=1., a=1.01*CLHEP::g/CLHEP::mole);
    G4Element* C = new G4Element("C", "C", z=6., a=12.01*CLHEP::g/CLHEP::mole);

    G4Material* mat = new G4Material("Glass", density=1.032*CLHEP::g/CLHEP::cm3,nelements=2);
    mat->AddElement(C,91.533*CLHEP::perCent);
    mat->AddElement(H,8.467*CLHEP::perCent);

    return mat ;
}

G4OpticalSurface* X4SurfaceTest::MakeOpticalSurface(const char* name)
{
    G4OpticalSurface* _surf = new G4OpticalSurface(name); 

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable ; 

    _surf->SetMaterialPropertiesTable( mpt ); 

    G4double pp[] = {2.0*eV, 3.5*eV};
    const G4int num = sizeof(pp)/sizeof(G4double);
    G4double reflectivity[] = {1., 1.};
    assert(sizeof(reflectivity) == sizeof(pp));
    mpt->AddProperty("REFLECTIVITY",pp,reflectivity,num);

    return _surf ; 
}



/**

It is crazy to have a separate lv for every module but the aim of this test 
is to stress the surface handling so lets do this crazy thing.

**/
void X4SurfaceTest::init()
{
    LOG(info) << "[ num_module " << num_module  ; 
    unsigned count = 0 ; 

    for(int ix=-nx ; ix <= nx ; ix++) 
    for(int iy=-ny ; iy <= ny ; iy++) 
    for(int iz=-nz ; iz <= nz ; iz++)
    {
        std::string lv_name = MakeName( "lv_mod", ix, iy, iz ); 
        std::string pv_name = MakeName( "pv_mod", ix, iy, iz ); 
        unsigned idx = Index(ix,iy,iz); 

        if( count % 1000 == 0 )
        {
            LOG(info) 
                << " count " << count 
                << " ix " << ix 
                << " iy " << iy 
                << " iz " << iz 
                << " idx " << idx 
                << " lv_name " << lv_name
                << " pv_name " << pv_name
                ;
        }

        G4LogicalVolume* lv = new G4LogicalVolume(so_module, glass, lv_name.c_str(), 0,0,0 ) ;  

        double x = double(ix)*1000. ; 
        double y = double(iy)*1000. ; 
        double z = double(iz)*1000. ; 

        G4PVPlacement* pv = new G4PVPlacement(0,G4ThreeVector(x,y,z),lv, pv_name.c_str(),lv_world,false,0);

        lv_module[idx] = lv ;  
        pv_module[idx] = pv ;

        std::string ss_surface_name = MakeName("ss_surface", ix, iy, iz ); 
        G4OpticalSurface* ss_surface = MakeOpticalSurface( ss_surface_name.c_str() );  
        std::string ss_name = MakeName("ss", ix, iy, iz ); 
        G4LogicalSkinSurface* ss = new G4LogicalSkinSurface( ss_name.c_str(), lv, ss_surface ) ; 

        std::string bs_surface_name = MakeName("bs_surface", ix, iy, iz ); 
        G4OpticalSurface* bs_surface = MakeOpticalSurface( bs_surface_name.c_str() );         
        std::string bs_name = MakeName("bs", ix, iy, iz ); 
        G4LogicalBorderSurface* bs = new G4LogicalBorderSurface(bs_name.c_str(), pv_world, pv, bs_surface );     

        ss_module[idx] = ss ; 
        bs_module[idx] = bs ;

        count += 1 ;  

    } 
    assert( count == num_module ); 
    LOG(info) << "]" ; 
}



std::string X4SurfaceTest::MakeName(const char* prefix, int ix, int iy, int iz)
{
    std::stringstream ss ; 
    ss << prefix 
       << "_" << ( ix < 0 ? 'n' : 'p' )  << std::setfill('0') << std::setw(4) << std::abs(ix)    
       << "_" << ( iy < 0 ? 'n' : 'p' )  << std::setfill('0') << std::setw(4) << std::abs(iy)    
       << "_" << ( iz < 0 ? 'n' : 'p' )  << std::setfill('0') << std::setw(4) << std::abs(iz)    
       ;  
    return ss.str(); 
}


int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv); 

    X4SurfaceTest t ; 

    const char* argforce = nullptr ; 

    LOG(info) << "[ convert " ; 
    GGeo* gg = X4PhysicalVolume::Convert(t.pv_world, argforce); 
    LOG(info) << "] convert " ; 

    assert( gg ); 

    return 0 ; 
}

