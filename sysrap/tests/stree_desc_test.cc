#include "stree.h"
#include "NPX.h"

int main(int argc, char** argv)
{
    const char* ss = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ; 

    stree t ; 
    int rc = t.load(ss); 
    if( rc != 0 ) return rc ; 

    //std::cout << "t.desc" << std::endl << t.desc() << std::endl ; 

    std::cout << "t.material.desc" << std::endl << t.material->desc() << std::endl ; 
    std::cout << "t.desc_mt" << std::endl << t.desc_mt() << std::endl; 

    std::cout << "t.surface.desc" << std::endl << t.surface->desc() << std::endl ; 
    std::cout << "t.desc_bd" << std::endl << t.desc_bd() << std::endl; 

    return 0 ; 
}
