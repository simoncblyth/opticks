
#include "OPTICKS_LOG.hh"
#include "spath.h"
#include "U4GDML.h"

#include "G4Material.hh"

struct Traverse
{
    const G4VPhysicalVolume* const world ;
    int count ;
    Traverse(const G4VPhysicalVolume* const world) ;
    void traverse_r(const G4VPhysicalVolume* const pv, int depth ) ;
    void visit( const G4VPhysicalVolume* const pv, int depth  );
};

inline Traverse::Traverse(const G4VPhysicalVolume* const world_)
    :
    world(world_),
    count(0)
{
    traverse_r(world, 0) ;
}

inline void Traverse::traverse_r(const G4VPhysicalVolume* const pv, int depth )
{
    visit( pv, depth  );
    count += 1 ;
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ;i++ ) traverse_r( lv->GetDaughter(i), depth+1  );
}

inline void Traverse::visit( const G4VPhysicalVolume* const pv, int depth  )
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    G4Material* mt = lv->GetMaterial() ;
    if( count % 10000 == 0 ) std::cout
        << "Traverse::visit"
        << " count " << count
        << " depth " << depth
        << " mt " << mt->GetName()
        << std::endl
        ;
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* path = spath::Resolve("$CFBaseFromGEOM/origin.gdml") ;

    LOG(info)
        << " argv[0] "
        << argv[0]
        << " path "
        << path ;

    new U4SensitiveDetector("PMTSDMgr") ;

    const G4VPhysicalVolume* world = U4GDML::Read(path) ;

    LOG_IF(fatal, world == nullptr)
        << " argv[0] " << argv[0] << "\n"
        << " path " << ( path ? path : "-" ) << "\n"
        << " world " << ( world ?  "YES" : "NO " ) << "\n"
        ;

    if(world == nullptr) return 0;

    Traverse trv(world);

    return 0 ;
}
