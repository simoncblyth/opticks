#include "s_csg.h"

int main()
{
    s_csg::Load("$BASE");  

    int lvid = ssys::getenvint("LVID",108) ; 

    std::vector<sn*> nds ; 
    s_csg::FindLVID( nds, lvid ); 
    std::cout << s_csg::Desc(nds) ; 

    sn* root = s_csg::FindLVIDRoot(lvid ); 

    std::cout << " root " << std::endl << ( root ? root->desc() : "-" ) << std::endl ; 

    return 0 ; 
}
