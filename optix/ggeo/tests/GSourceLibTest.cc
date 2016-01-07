//  ggv --gsrclib
//  ggv --gsrclib --debug
//

#include "Opticks.hh"

#include "GCache.hh"
#include "GSource.hh"
#include "GSourceLib.hh"

#include "NPY.hpp"


int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv, "gsrclib.log");

    GCache* cache = new GCache(opticks);

    GSourceLib* sl = new GSourceLib(cache);

    sl->generateBlackBodySample();

    //GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );    
    //sl->add(source);

    //NPY<float>* buf = sl->createBuffer();

    //buf->save("/tmp/gsrclib.npy");

    return 0 ; 
}

