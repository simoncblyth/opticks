#include "stree.h"

int main(int argc, char** argv)
{
    stree* t = stree::Load();
    if(!t) return 1 ;

    //std::cout << "t.desc" << std::endl << t->desc() << std::endl ; 

    std::cout << "t.material.desc" << std::endl << t->material->desc() << std::endl ; 
    std::cout << "t.desc_mt" << std::endl << t->desc_mt() << std::endl; 

    std::cout << "t.surface.desc" << std::endl << t->surface->desc() << std::endl ; 
    std::cout << "t.desc_bd" << std::endl << t->desc_bd() << std::endl; 

    t->save_desc("$FOLD");


    return 0 ; 
}
