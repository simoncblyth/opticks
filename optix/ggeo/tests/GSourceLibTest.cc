//  ggv --gsrclib
//  ggv --gsrclib --debug
//

#include "GCache.hh"
#include "GSource.hh"
#include "GSourceLib.hh"

#include "NPY.hpp"


int main(int argc, char** argv)
{
    GCache* cache = new GCache("GGEOVIEW_", "gsrclib.log", "info");

    cache->configure(argc, argv);

    GSourceLib* sl = new GSourceLib(cache);

    sl->generateBlackBodySample();

    //GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );    
    //sl->add(source);

    //NPY<float>* buf = sl->createBuffer();

    //buf->save("/tmp/gsrclib.npy");

    return 0 ; 
}

