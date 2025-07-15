
#include <sstream>
#include "OPTICKS_LOG.hh"
#include "U4VolumeMaker.hh"
#include "ssys.h"

#include "G4Material.hh"
#include "G4VSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "U4Volume.h"



struct U4VolumeMakerTest
{
    static std::string Desc(const G4VPhysicalVolume* pv, const char* symbol );
    static std::string Desc(const G4VSolid* so, const char* symbol);
    std::string desc() const ;

    const char*              sub ;
    const G4VPhysicalVolume* pv ;
    const G4VPhysicalVolume* spv ;
    const G4LogicalVolume*   slv ;

    const G4VPhysicalVolume* spv1 ;
    const G4VPhysicalVolume* spv2 ;

    const G4LogicalVolume* slv1 ;
    const G4LogicalVolume* slv2 ;

    const G4VSolid* sso1 ;
    const G4VSolid* sso2 ;

    U4VolumeMakerTest();

    int sub_check();
    int OldHamaBodyPhys();
};



std::string U4VolumeMakerTest::Desc(const G4VPhysicalVolume* pv, const char* symbol)
{
    const G4LogicalVolume* lv = pv ? pv->GetLogicalVolume() : nullptr ;
    const G4Material* mt = lv ? lv->GetMaterial() : nullptr ;
    const G4VSolid* so = lv ? lv->GetSolid() : nullptr ;
    std::stringstream ss ;
    ss
        << symbol
        << " pv " << ( pv ? pv->GetName() : "-" )
        << " lv " << ( lv ? lv->GetName() : "-" )
        << " mt " << ( mt ? mt->GetName() : "-" )
        << " so " << ( so ? so->GetName() : "-" )
        << "\n"
        ;
    std::string str = ss.str();
    return str ;
}

std::string U4VolumeMakerTest::Desc(const G4VSolid* so, const char* symbol )
{
    G4GeometryType et = so ? so->GetEntityType() : "" ;
    std::stringstream ss ;
    ss
        << symbol
        << " so " << ( so ? so->GetName() : "-" )
        << " et " << et
        << "\n"
        ;

    std::string str = ss.str();
    return str ;
}



std::string U4VolumeMakerTest::desc() const
{
    std::stringstream ss ;
    ss << U4VolumeMaker::Desc()
       << " sub " << ( sub ? sub : "-" )
       << Desc(pv, "pv")
       << Desc(spv, "spv")
       << Desc(spv1, "spv1")
       << Desc(spv2, "spv2")
       << Desc(sso1, "sso1")
       << Desc(sso2, "sso2")
       ;

    std::string str = ss.str() ;
    return str ;
}


inline U4VolumeMakerTest::U4VolumeMakerTest()
    :
    sub(ssys::getenvvar("SUB", "hama_body_phys")),
    pv(U4VolumeMaker::PV()),
    spv(pv ? U4Volume::FindPV(pv, sub ) : nullptr),
    slv(spv ? spv->GetLogicalVolume() : nullptr),
    spv1(slv ? slv->GetDaughter(0) : nullptr ),
    spv2(slv ? slv->GetDaughter(1) : nullptr ),
    slv1(spv1 ? spv1->GetLogicalVolume() : nullptr ),
    slv2(spv2 ? spv2->GetLogicalVolume() : nullptr ),
    sso1(slv1 ? slv1->GetSolid() : nullptr ),
    sso2(slv2 ? slv2->GetSolid() : nullptr )
{
    std::cout << desc();
}

inline int U4VolumeMakerTest::sub_check()
{
    int rc = 0 ;
    if(sub && strcmp(sub, "hama_body_phys") == 0)  rc += OldHamaBodyPhys();
    return rc ;
}

inline int U4VolumeMakerTest::OldHamaBodyPhys()
{
    if(spv == nullptr) return 0 ;
    LOG(info) << " spv :: " << Desc(spv, "spv")  ;
    LOG(info) << " spv1 :: " << Desc(spv1, "spv1")  ;
    LOG(info) << " spv2 :: " << Desc(spv2, "spv2")  ;

    if( spv1 == nullptr || spv2 == nullptr ) return 0 ;

    assert( sso1 );

    G4ThreeVector pos(103,0,-50) ;
    G4ThreeVector dir(0,0,1) ;  // +Z

    G4double dist1 = sso1->DistanceToOut(pos, dir); // farside
    G4double dist2 = sso2->DistanceToIn(pos, dir);  // NOT GETTING WHAT I EXPECTED WITH A UNION OF 2 POLYCONE AND ELLIPSOID

    std::cout << " dist1 = sso1->DistanceToOut(pos, dir)  : " << dist1 << std::endl ;
    std::cout << " dist2 = sso2->DistanceToIn( pos, dir)  : " << dist2 << std::endl ;

    std::cout << Desc(sso1, "sso1") ;
    std::cout << Desc(sso2, "sso2") ;

    return 0 ;
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    U4VolumeMakerTest test ;
    return test.sub_check() ;
}
