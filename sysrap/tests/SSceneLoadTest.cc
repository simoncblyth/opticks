/**
SceneLoadTest.cc
==================

::

    ~/o/sysrap/tests/SSceneLoadTest.sh

**/

#include "ssys.h"
#include "SScene.h"


struct SSceneLoadTest
{
    static int Load_();
    static int desc();
    static int descDetail();
    static int findSubMesh();
    static int Main();
};

inline int SSceneLoadTest::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "Load_");
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"Load_")) rc += Load_();
    if(ALL||0==strcmp(TEST,"desc"))  rc += desc();
    if(ALL||0==strcmp(TEST,"descDetail"))  rc += descDetail();
    if(ALL||0==strcmp(TEST,"findSubMesh"))  rc += findSubMesh();
    return rc ;
}

inline int SSceneLoadTest::Load_()
{
    SScene* a = SScene::Load_() ;
    if( a == nullptr ) return 0 ;

    const SBitSet* elv = SGeoConfig::ELV(a->id);
    SScene* b = a->copy(elv);

    int mismatch = SScene::Compare(a, b) ;
    if( mismatch > 0 ) std::cout
        << "[A:\n"
        << a->desc()
        << "]A\n"
        << "[B:\n"
        << b->desc()
        << "]B:\n"
        << "[AB:\n"
        << SScene::DescCompare(a,b)
        << "]AB:\n"
        ;

    std::cout << "SceneLoadTest.main mismatch " << mismatch << "\n" ;

    return mismatch ;
}

inline int SSceneLoadTest::desc()
{
    SScene* a = SScene::Load() ;
    if(!a) return 1 ;
    std::cout << a->desc();
    return 0 ;
}

inline int SSceneLoadTest::descDetail()
{
    SScene* a = SScene::Load() ;
    if(!a) return 1 ;
    std::cout << a->descDetail();
    return 0 ;
}

inline int SSceneLoadTest::findSubMesh()
{
    SScene* a = SScene::Load() ;
    if(!a) return 1 ;

    int LVID = ssys::getenvint("LVID", 0);
    std::vector<const SMesh*> subs ;

    std::cout
       << "[SSceneLoadTest::findSubMesh"
       << " LVID " << LVID
       << "\n"
       ;


    a->findSubMesh(subs, LVID);

    std::cout
       << "]SSceneLoadTest::findSubMesh"
       << " LVID " << LVID
       << " subs.size " << subs.size()
       << "\n"
       << "[SScene::descSubMesh\n"
       << a->descSubMesh(subs)
       << "]SScene::descSubMesh\n"
       ;

    return 0 ;
}


int main()
{
    return SSceneLoadTest::Main() ;
}
