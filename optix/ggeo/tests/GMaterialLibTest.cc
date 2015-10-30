#include "GCache.hh"
#include "GMaterialLib.hh"

int main()
{
    GCache gc("GGEOVIEW_");

    GMaterialLib* lib = GMaterialLib::load(&gc);

    const char* mats = "Acrylic,GdDopedLS,LiquidScintillator,ESR,MineralOil" ;

    lib->dumpMaterials(mats);

    return 0 ;
}

