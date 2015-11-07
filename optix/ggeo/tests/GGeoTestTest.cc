//  ggv --geotest

#include "GCache.hh"
#include "GGeoTest.hh"


void test_geotest(GCache* cache, const char* config)
{
    GGeoTest* geotest = new GGeoTest(cache);
    geotest->configure(config);
    geotest->dump();
    geotest->modifyGeometry();
}

void test_bib(GCache* cache)
{
    const char* config = 
    "mode=BoxInBox;"
    "dimensions=4,2,0,0;"
    "boundary=MineralOil/Rock/perfectAbsorbSurface/;"
    "boundary=Pyrex/MineralOil//;"
    ;
    test_geotest(cache, config);
}

int main(int argc, char** argv)
{
    GCache* cache = new GCache("GGEOVIEW_", "geotest.log", "info");
    cache->configure(argc, argv);

    test_bib(cache);

    return 1 ;
}
