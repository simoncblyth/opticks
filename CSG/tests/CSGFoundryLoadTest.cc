/**

~/o/CSG/tests/CSGFoundryLoadTest.sh

**/


#include "OPTICKS_LOG.hh"
#include "SSim.hh"
#include "stree.h"
#include "ssys.h"
#include "SScene.h"
#include "CSGFoundry.h"

struct CSGFoundryLoadTest
{
    static int Load();
    static int getMeshPrim();
    static int descPrimRange();
    static int CompareRanges();

    static int Main(int argc, char** argv);
};


inline int CSGFoundryLoadTest::Main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* TEST = ssys::getenvvar("TEST", "Load");
    bool ALL = strcmp(TEST,"ALL") == 0 ;
    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"Load")) rc += Load();
    if(ALL||0==strcmp(TEST,"getMeshPrim")) rc += getMeshPrim();
    if(ALL||0==strcmp(TEST,"descPrimRange")) rc += descPrimRange();
    if(ALL||0==strcmp(TEST,"CompareRanges")) rc += CompareRanges();
    return rc ;
}

inline int CSGFoundryLoadTest::Load()
{
    //SSim::Create() ; // SEEMS NO LONGER NEEDED ?
    CSGFoundry* cf = CSGFoundry::Load() ;

    LOG(info) << " -------------------- After CSGFoundry::Load " ;

    LOG(info) << cf->desc() ;
    LOG(info) << " -------------------- After CSGFoundry::desc " ;

    stree* st = cf->sim->tree ;
    LOG(info) << st->desc() ;
    LOG(info) << " -------------------- After stree::desc " ;

    return 0 ;
}

inline int CSGFoundryLoadTest::getMeshPrim()
{
    int LVID = ssys::getenvint("LVID", 0);
    std::cout << "[CSGFoundryLoadTest::getMeshPrim LVID " << LVID << "\n" ;
    CSGFoundry* fd = CSGFoundry::Load() ;

    unsigned mesh_idx = LVID ;
    std::vector<const CSGPrim*> prim ;
    fd->getMeshPrimPointers(prim, mesh_idx);

    std::cout << "]CSGFoundryLoadTest::getMeshPrim LVID " << LVID << "\n" << CSGPrim::Desc(prim) ;

    return 0 ;
}

inline int CSGFoundryLoadTest::descPrimRange()
{
    std::cout << "[CSGFoundryLoadTest::descPrimRange \n" ;
    CSGFoundry* fd = CSGFoundry::Load() ;
    std::cout << fd->descPrimRange() ;
    std::cout << "]CSGFoundryLoadTest::descPrimRange\n" ;
    return 0 ;
}

inline int CSGFoundryLoadTest::CompareRanges()
{
    CSGFoundry* fd = CSGFoundry::Load() ;
    size_t num_solid = fd->solid.size() ;
    int solidIdx = num_solid - 1 ;
    stree* tr = fd->sim->tree ;
    SScene* sc = fd->sim->scene ;

    assert( sc->meshgroup.size() == num_solid );

    const SMeshGroup* mg = sc->meshgroup[solidIdx] ;

    std::cout
        << "[CSGFoundryLoadTest::CompareRanges \n"
        << " num_solid " << num_solid
        << " solidIdx " << solidIdx
        << " tr " << ( tr ? "YES" : "NO " )
        << " sc " << ( sc ? "YES" : "NO " )
        << " mg " << ( mg ? "YES" : "NO " )
        << "\n"
        ;

    std::cout << fd->comparePrimRange(solidIdx, mg) ;

    std::cout << "]CSGFoundryLoadTest::CompareRanges\n" ;
    return 0 ;
}


int main(int argc, char** argv)
{
    return CSGFoundryLoadTest::Main(argc, argv) ;
}
