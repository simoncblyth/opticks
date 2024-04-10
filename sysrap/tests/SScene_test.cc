/**
SScene_test.cc
================

::

   ~/o/sysrap/tests/SScene_test.sh 
   ~/o/sysrap/tests/SScene_test.cc

**/

#include "ssys.h"
#include "SScene.h"

struct SScene_test
{
    static int CreateFromTree(); 
    static int Load(); 
    static int Main();
}; 

inline int SScene_test::CreateFromTree()
{
    std::cout << "[SScene_test::CreateFromTree" << std::endl ; 
    stree* st = stree::Load("$TREE_FOLD"); 
    std::cout << st->desc() ; 

    SScene scene ; 
    scene.initFromTree(st); 
    
    std::cout << scene.desc() ; 
    scene.save("$SCENE_FOLD") ;  // "scene" reldir is implicit 

    std::cout << "]SScene_test::CreateFromTree" << std::endl ; 
    return 0 ; 
}

inline int SScene_test::Load()
{
    std::cout << "[SScene_test::Load" << std::endl ; 
    SScene* scene = SScene::Load("$SCENE_FOLD") ; 
    std::cout << scene->desc() ; 
    std::cout << "]SScene_test::Load" << std::endl ; 
    return 0 ; 
}

inline int SScene_test::Main()
{
    int rc(0) ; 
    const char* TEST = ssys::getenvvar("TEST", "CreateFromTree"); 
  
    if(       strcmp(TEST,"CreateFromTree") == 0 ) rc += CreateFromTree() ; 
    else if ( strcmp(TEST,"Load") == 0 )           rc += Load() ;   

    return rc ;  
}

int main()
{
    return SScene_test::Main();
}

