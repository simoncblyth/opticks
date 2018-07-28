#include "OPTICKS_LOG.hh"
#include "X4CSG.hh"
#include "X4Solid.hh"
#include "NCSG.hpp"
#include "NCSGData.hpp"
#include "NCSGList.hpp"
#include "NNode.hpp"


// start of portion to be generated ----------------
#include "G4Box.hh"
#include "G4Orb.hh"
#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"
#include "G4UnionSolid.hh"
#include "G4VisExtent.hh"

G4VSolid* make_solid()
{
    G4VSolid* b = new G4Orb("orb",10) ;
    G4VSolid* d = new G4Box("box",7,7,7) ;
    G4RotationMatrix* A = new G4RotationMatrix(G4ThreeVector(0.707107,-0.707107,0.000000),G4ThreeVector(0.707107,0.707107,0.000000),G4ThreeVector(0.000000,0.000000,1.000000));
    G4ThreeVector B(1.000000,0.000000,0.000000);
    G4VSolid* a = new G4UnionSolid("uni1",b , d , A , B) ;
    return a ; 
}
// end of portion to be generated ---------------------


G4VSolid* make_container(G4VSolid* so, float scale=5.f )
{
    G4VisExtent ve = so->GetExtent();

    float xmin = ve.GetXmin() ;
    float ymin = ve.GetYmin() ;
    float zmin = ve.GetZmin() ;

    float xmax = ve.GetXmax() ;
    float ymax = ve.GetYmax() ;
    float zmax = ve.GetZmax() ;

    float hx = scale*std::max(std::abs(xmin),std::abs(xmax)) ; 
    float hy = scale*std::max(std::abs(ymin),std::abs(ymax)) ; 
    float hz = scale*std::max(std::abs(zmin),std::abs(zmax)) ; 

    G4VSolid* cso = new G4Box("container", hx, hy, hz ); 

    return cso ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    X4CSG yc ;  

    G4VSolid* so_ = make_solid() ; 
    G4VSolid* co_ = make_container(so_) ; 

    std::cout << "so_" << std::endl << *so_ << std::endl ; 
    std::cout << "co_" << std::endl << *co_ << std::endl ; 

    nnode* so = X4Solid::Convert(so_) ; assert( so ) ; 
    nnode* co = X4Solid::Convert(co_) ; assert( co ) ; 

    co->boundary = "Rock//perfectAbsorbSurface/Vacuum" ; 
    so->boundary = "Vacuum///GlassSchottF2" ; 

    NCSG* so_csg = NCSG::Adopt( so ); assert( so_csg ) ;  
    NCSG* co_csg = NCSG::Adopt( co ); assert( co_csg ) ;  

 
    NCSGData* so_data = so_csg->getCSGData();
    so_data->setMeta<std::string>( "poly", "IM" );  
    so_data->setMeta<std::string>( "resolution", "20" );  

    NCSGData* co_data = co_csg->getCSGData();
    co_data->setMeta<std::string>( "poly", "IM" );  
    co_data->setMeta<std::string>( "resolution", "20" );  

    std::vector<NCSG*> trees = {co_csg, so_csg };

    const char* csgpath = "$TMP/X4CSGTest" ; 

    unsigned verbosity = 1 ; 

    NCSGList* ls = NCSGList::Create( trees, csgpath , verbosity ); 

    ls->savesrc(); 

    // NCSGList* ls2 = NCSGList::Load( csgpath , verbosity );   // see whats missing from the save 

    // NB only stderr emission 
    std::cerr << "analytic=1_csgpath=" << csgpath << std::endl ; 

    return 0 ; 
}


