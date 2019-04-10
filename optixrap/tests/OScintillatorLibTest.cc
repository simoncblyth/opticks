
#include "OScintillatorLib.hh"
#include "Opticks.hh"
#include "GScintillatorLib.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);

    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();

    optix::Context context = optix::Context::create();

    OScintillatorLib* oscin ;  
    oscin = new OScintillatorLib(context, slib );

    const char* slice = "0:1" ; 
    oscin->convert(slice);

    LOG(info) << "DONE"  ;

    return 0 ; 
}


