#include "SScene.h"

int main()
{
    SScene* scene = SScene::Load("$SCENE_FOLD") ; 
    std::cout << scene->desc() ; 
 
    return 0 ; 
}
