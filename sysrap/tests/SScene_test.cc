/**
SScene_test.cc
================

::

   ~/o/sysrap/tests/SScene_test.sh 

**/

#include "SScene.h"

struct SScene_test
{
    static constexpr const char* SCENE_DIR = "$SCENE_FOLD/scene" ; 
    static int CreateFromTree(); 
    static int Load(); 
    static int Main();
}; 

inline int SScene_test::CreateFromTree()
{
    stree* st = stree::Load("$TREE_FOLD"); 
    std::cout << st->desc() ; 

    SScene scene ; 
    scene.initFromTree(st); 

    std::cout << scene.desc() ; 
    scene.save(SCENE_DIR) ;  

    return 0 ; 
}

inline int SScene_test::Load()
{
    SScene* scene = SScene::Load(SCENE_DIR) ; 
    std::cout << scene->desc() ; 
    return 0 ; 
}

inline int SScene_test::Main()
{
    int rc(0) ; 
    //rc += CreateFromTree() ; 
    rc += Load() ; 
    return rc ;  
}

int main()
{
    return SScene_test::Main();
}

