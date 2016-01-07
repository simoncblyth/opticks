//  ggv --geotest

#include "Opticks.hh"
#include "GCache.hh"

#include "GGeoTestConfig.hh"
#include "GGeoTest.hh"


void test_geotest(GCache* cache, const char* config)
{
    GGeoTestConfig* gtc = new GGeoTestConfig(config);
    GGeoTest* geotest = new GGeoTest(cache, gtc);
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

void test_config()
{
    const char* config = 
    "mode=BoxInBox;"
    "dimensions=4,2,0,0;"
    "boundary=MineralOil/Rock/perfectAbsorbSurface/;"
    "boundary=Pyrex/MineralOil//;"
    ;

    GGeoTestConfig* gtc = new GGeoTestConfig(config);
    gtc->dump();
}


int main(int argc, char** argv)
{
    test_config();

    Opticks* opticks = new Opticks(argc, argv, "geotest.log");
    GCache* cache = new GCache(opticks);

    test_bib(cache);

    return 1 ;
}
