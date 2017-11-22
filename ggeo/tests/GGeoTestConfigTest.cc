
#include "NGLM.hpp"
#include "GGeoTestConfig.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_BODY.hh"

const char* CONFIG = 
    "mode=BoxInBox_"
    "node=box_"
    "boundary=MineralOil/Rock/perfectAbsorbSurface/_"
    "parameters=-1,1,0,500_"
    "node=box_"
    "boundary=Pyrex/MineralOil//_"
    "parameters=-1,1,0,100_"
    "autocontainer=Rock//perfectAbsorbSurface/Vacuum_"
    "autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_"
    ;

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    GGeoTestConfig* gtc = new GGeoTestConfig(CONFIG);
    gtc->dump();


    return 0 ;
}
