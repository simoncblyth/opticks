
#include <sstream>
#include "OPTICKS_LOG.hh"
#include "U4VolumeMaker.hh"
#include "ssys.h"

#include "G4Material.hh"
#include "G4VSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "U4Volume.h"

std::string Desc(const G4VPhysicalVolume* pv)
{
    const G4LogicalVolume* lv = pv ? pv->GetLogicalVolume() : nullptr ; 
    const G4Material* mt = lv ? lv->GetMaterial() : nullptr ; 
    const G4VSolid* so = lv ? lv->GetSolid() : nullptr ; 
    std::stringstream ss ; 
    ss 
        << " pv " << ( pv ? pv->GetName() : "-" )
        << " lv " << ( lv ? lv->GetName() : "-" )
        << " mt " << ( mt ? mt->GetName() : "-" )
        << " so " << ( so ? so->GetName() : "-" )
        ;
    std::string str = ss.str(); 
    return str ; 
}

std::string Desc(const G4VSolid* so )
{
    G4GeometryType et = so->GetEntityType()  ; 
    std::stringstream ss ; 
    ss 
        << " so " << ( so ? so->GetName() : "-" )
        << " et " << et 
        ;
 
    std::string str = ss.str(); 
    return str ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info) << U4VolumeMaker::Desc() ; 

    const G4VPhysicalVolume* pv = U4VolumeMaker::PV();  
    if(pv == nullptr) return 0 ; 
    LOG(info) << " pv :: " << Desc(pv) ;  

    const char* sub = ssys::getenvvar("SUB", "hama_body_phys") ; 
    LOG(info) << " sub " << sub ; 

    const G4VPhysicalVolume* spv = U4Volume::FindPV(pv, sub ); 
    const G4LogicalVolume* slv = spv ? spv->GetLogicalVolume() : nullptr ; 
    LOG(info) << " spv :: " << Desc(spv)  ;  
    if(spv == nullptr) return 0 ; 

    const G4VPhysicalVolume* spv1 = slv ? slv->GetDaughter(0) : nullptr ; 
    const G4VPhysicalVolume* spv2 = slv ? slv->GetDaughter(1) : nullptr ; 
    LOG(info) << " spv1 :: " << Desc(spv1)  ;  
    LOG(info) << " spv2 :: " << Desc(spv2)  ;  

    if( spv1 == nullptr || spv2 == nullptr ) return 0 ; 

    const G4LogicalVolume* slv1 = spv1 ? spv1->GetLogicalVolume() : nullptr ; 
    const G4LogicalVolume* slv2 = spv2 ? spv2->GetLogicalVolume() : nullptr ; 

    const G4VSolid* sso1 = slv1 ? slv1->GetSolid() : nullptr ; 
    const G4VSolid* sso2 = slv2 ? slv2->GetSolid() : nullptr ; 

    assert( sso1 ); 

    G4ThreeVector pos(103,0,-50) ; 
    G4ThreeVector dir(0,0,1) ;  // +Z

    G4double dist1 = sso1->DistanceToOut(pos, dir); // farside 
    G4double dist2 = sso2->DistanceToIn(pos, dir);  // NOT GETTING WHAT I EXPECTED WITH A UNION OF 2 POLYCONE AND ELLIPSOID

    std::cout << " dist1 = sso1->DistanceToOut(pos, dir)  : " << dist1 << std::endl ; 
    std::cout << " dist2 = sso2->DistanceToIn( pos, dir)  : " << dist2 << std::endl ; 

    std::cout << " sso1 " << Desc(sso1) << std::endl ; 
    std::cout << " sso2 " << Desc(sso2) << std::endl ; 

    return 0 ; 
}
