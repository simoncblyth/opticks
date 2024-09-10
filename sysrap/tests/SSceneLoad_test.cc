/**
SceneLoad_test.cc
==================

::

    ~/o/sysrap/tests/SSceneLoad_test.sh 


**/

#include "SScene.h"

int main()
{
    SScene* a = SScene::Load("$SCENE_FOLD") ; 

    SScene* b = a->copy(nullptr); 

    //std::cout << "A:" << a->desc() ; 
    //std::cout << "B:" << b->desc() ; 


    int mismatch = SScene::Compare(a, b) ; 

    if( mismatch > 0 ) std::cout << SScene::DescCompare(a,b);  

    std::cout << "SceneLoad_test.main mismatch " << mismatch << "\n" ; 

    return mismatch ; 
}
