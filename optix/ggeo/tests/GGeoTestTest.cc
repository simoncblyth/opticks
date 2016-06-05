//  ggv --geotest

#include "Opticks.hh"
#include "GGeoTestConfig.hh"
#include "GGeoTest.hh"


void test_geotest(Opticks* opticks, const char* config)
{
    GGeoTestConfig* gtc = new GGeoTestConfig(config);
    GGeoTest* geotest = new GGeoTest(opticks, gtc);
    geotest->dump();
    geotest->modifyGeometry();
}

void test_bib(Opticks* opticks)
{
    const char* config = 
    "mode=BoxInBox;"
    "dimensions=4,2,0,0;"
    "boundary=MineralOil/Rock/perfectAbsorbSurface/;"
    "boundary=Pyrex/MineralOil//;"
    ;

    test_geotest(opticks, config);
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

    test_bib(opticks);

    return 1 ;
}
