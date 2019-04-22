//  ggv --gsrclib
//  ggv --gsrclib --debug
//

#include "Opticks.hh"

#include "GSource.hh"
#include "GSourceLib.hh"

#include "NPY.hpp"

#include "OPTICKS_LOG.hh"
#include "GGEO_BODY.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks* ok = new Opticks(argc, argv);
    ok->configure(); 

    GSourceLib* sl = new GSourceLib(ok);

    sl->generateBlackBodySample();

    GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );    
    sl->add(source);

    NPY<float>* buf = sl->createBuffer();

    buf->save("$TMP/gsrclib.npy");

    return 0 ; 
}

