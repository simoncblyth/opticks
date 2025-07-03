/**
SceneLoadTest.cc
==================

::

    ~/o/sysrap/tests/SSceneLoadTest.sh

**/

#include "SScene.h"

int main()
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
