#include "stree.h"

int main(int argc, char** argv)
{
    stree* t = stree::Load();
    if(!t) return 1 ;

    //std::cout << "t.desc" << std::endl << t->desc() << std::endl ;

    std::cout << t->desc_solids() << "\n" ;

   /*
    std::cout
        << "[t.material.desc\n"
        << t->material->desc()
        << "]t.material.desc\n"
        << "[t.desc_mt\n"
        << t->desc_mt()
        << "]t.desc_mt\n"
        << "[t.surface.desc\n"
        << t->surface->desc()
        << "[t.surface.desc\n"
        << "[t.desc_bd\n"
        << t->desc_bd()
        << "]t.desc_bd\n"
        ;
    */

    t->save_desc("$FOLD");


    return 0 ;
}
