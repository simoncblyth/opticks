#include "ssys.h"

#include "G4VSolid.hh"

#include "U4Mesh.h"
#include "U4Solid.h"
#include "U4SolidMaker.hh"

#include "sn.h"

int main()
{
    const char* GEOM = ssys::getenvvar("GEOM", "WaterDistributer");
    bool can_make = U4SolidMaker::CanMake(GEOM);
    if(!can_make) return 0 ;

    const G4VSolid* solid = U4SolidMaker::Make(GEOM);

    int lvid = 0 ;
    int depth = 0 ;
    int level = 4 ;

    sn* nd = U4Solid::Convert(solid, lvid, depth, level);
    std::cout
        << "[U4SolidMakerTest nd.desc\n"
        <<   nd->desc()
        << "]U4SolidMakeTest nd.desc\n"
        ;

    if(level > 2 ) std::cout
        << "\n[U4SolidMakerTest nd->render() \n"
        << nd->render()
        << "\n]U4SolidMakerTest nd->render() \n\n"
        ;

    if(level > 3 ) std::cout
        << "\n[U4SolidMakerTest nd->detail_r()\n"
        << nd->detail_r()
        << "\n]U4SolidMakerTest nd->detail_r() \n\n"
        ;

    if(level > 3 ) std::cout
        << "\n[U4SolidMakerTest  nd->desc_prim_all() \n"
        << nd->desc_prim_all(false)
        << "\n]U4SolidMakerTest  nd->desc_prim_all() \n"
        ;

    std::cout << "sn::Desc.0.before-delete-expect-some-nodes\n"  << sn::Desc() << "\n" ;
    delete nd ;
    std::cout << "sn::Desc.1.after-delete-expect-no-nodes\n"  << sn::Desc() << "\n" ;



    NPFold* fold = U4Mesh::Serialize(solid) ;
    fold->set_meta<std::string>("GEOM",GEOM);
    fold->set_meta<std::string>("desc","placeholder-desc");
    fold->save("$FOLD", GEOM );

    return 0;
}
