/**

~/o/u4/tests/U4SolidMakerTest.sh

**/


#include "ssys.h"
#include "G4VSolid.hh"
#include "U4Mesh.h"
#include "U4Solid.h"
#include "U4SolidMaker.hh"

#include "sn.h"


struct U4SolidMakerTest
{
    static const char* GEOM ;
    static const char* TEST ;
    static const int   LEVEL ;

    static const G4VSolid* MakeSolid();
    static sn* Convert_(const G4VSolid* solid);

    static int Convert();
    static int Main();
};

const char* U4SolidMakerTest::GEOM = ssys::getenvvar("GEOM", "WaterDistributer");
const char* U4SolidMakerTest::TEST = ssys::getenvvar("TEST", "Convert");
const int  U4SolidMakerTest::LEVEL = ssys::getenvint("LEVEL", 4 );

inline int U4SolidMakerTest::Main()
{
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"Convert")) rc += Convert();
    return rc ;
}

inline const G4VSolid* U4SolidMakerTest::MakeSolid()
{
    bool can_make = U4SolidMaker::CanMake(GEOM);
    if(!can_make) return 0 ;
    const G4VSolid* solid = U4SolidMaker::Make(GEOM);
    return solid ;
}

inline sn* U4SolidMakerTest::Convert_(const G4VSolid* solid )
{
    int lvid = 0 ;
    int depth = 0 ;
    sn* nd = U4Solid::Convert(solid, lvid, depth, LEVEL);
    return nd ;
}


inline int U4SolidMakerTest::Convert()
{
    const G4VSolid* solid = MakeSolid();
    NPFold* fold = U4Mesh::Serialize(solid) ;
    fold->set_meta<std::string>("GEOM",GEOM);
    fold->set_meta<std::string>("desc","placeholder-desc");
    fold->save("$FOLD", GEOM );

    sn* nd = Convert_(solid);

    std::cout
        << "[U4SolidMakerTest nd.desc\n"
        <<   nd->desc()
        << "]U4SolidMakeTest nd.desc\n"
        ;

    if(LEVEL > 2 ) std::cout
        << "\n[U4SolidMakerTest nd->render() \n"
        << nd->render()
        << "\n]U4SolidMakerTest nd->render() \n\n"
        ;

    if(LEVEL > 3 ) std::cout
        << "\n[U4SolidMakerTest nd->detail_r()\n"
        << nd->detail_r()
        << "\n]U4SolidMakerTest nd->detail_r() \n\n"
        ;

    if(LEVEL > 3 ) std::cout
        << "\n[U4SolidMakerTest  nd->desc_prim_all() \n"
        << nd->desc_prim_all(false)
        << "\n]U4SolidMakerTest  nd->desc_prim_all() \n"
        ;



    std::cout << "sn::Desc.0.before-delete-expect-some-nodes\n"  << sn::Desc() << "\n" ;
    delete nd ;
    std::cout << "sn::Desc.1.after-delete-expect-no-nodes\n"  << sn::Desc() << "\n" ;

    return 0;
}


int main()
{
    return U4SolidMakerTest::Main();
}
