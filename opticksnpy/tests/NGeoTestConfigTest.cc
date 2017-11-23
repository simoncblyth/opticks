
#include "NGLM.hpp"
#include "NGeoTestConfig.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"

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
    NPY_LOG__ ;

    NGeoTestConfig gtc(CONFIG);
    gtc.dump();


    return 0 ;
}
