//  ggv --geotest

#include "Opticks.hh"
#include "GGeoTestConfig.hh"
#include "GGeoTest.hh"


#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_BODY.hh"


const char* CONFIG = 
    "mode=BoxInBox_"
    "shape=box_"
    "boundary=MineralOil/Rock/perfectAbsorbSurface/_"
    "parameters=-1,1,0,500_"
    "shape=box_"
    "boundary=Pyrex/MineralOil//_"
    "parameters=-1,1,0,100_"
    ;



void test_geotest(Opticks* opticks, const char* config)
{
    GGeoTestConfig* gtc = new GGeoTestConfig(config);
    GGeoTest* geotest = new GGeoTest(opticks, gtc);
    geotest->dump();
    geotest->modifyGeometry();
}

void test_bib(Opticks* opticks)
{
    test_geotest(opticks, CONFIG);
}

void test_config()
{
    const char* config = CONFIG ; 
    GGeoTestConfig* gtc = new GGeoTestConfig(config);
    gtc->dump();
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    test_config();

    Opticks* opticks = new Opticks(argc, argv);

    test_bib(opticks);

    return 0 ;
}
