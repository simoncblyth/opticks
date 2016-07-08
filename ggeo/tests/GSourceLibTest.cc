//  ggv --gsrclib
//  ggv --gsrclib --debug
//

#include "Opticks.hh"

#include "GSource.hh"
#include "GSourceLib.hh"

#include "NPY.hpp"

#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_BODY.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    Opticks* opticks = new Opticks(argc, argv);

    GSourceLib* sl = new GSourceLib(opticks);

    sl->generateBlackBodySample();

    GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );    
    sl->add(source);

    NPY<float>* buf = sl->createBuffer();

    buf->save("$TMP/gsrclib.npy");

    return 0 ; 
}

