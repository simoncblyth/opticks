#include "OScintillatorLib.hh"
#include "Opticks.hh"
#include "GScintillatorLib.hh"

#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "OXRAP_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKCORE_LOG__ ; 
    GGEO_LOG__ ; 
    OXRAP_LOG__ ; 

    Opticks ok(argc, argv);

    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();

    optix::Context context = optix::Context::create();

    OScintillatorLib* oscin ;  
    oscin = new OScintillatorLib(context, slib );

    oscin->convert();


    return 0 ; 
}
