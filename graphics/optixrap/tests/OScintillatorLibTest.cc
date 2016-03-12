#include "OScintillatorLib.hh"
#include "Opticks.hh"
#include "GCache.hh"
#include "GScintillatorLib.hh"


int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv, "oscint.log");
    GCache gc(opticks);

    GScintillatorLib* slib = GScintillatorLib::load(&gc);
    slib->dump();

    optix::Context context = optix::Context::create();

    OScintillatorLib* oscin ;  
    oscin = new OScintillatorLib(context, slib );

    oscin->convert();


    return 0 ; 
}
