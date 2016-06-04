#include "OScintillatorLib.hh"
#include "Opticks.hh"
#include "GScintillatorLib.hh"


int main(int argc, char** argv)
{
    Opticks ok(argc, argv, "oscint.log");

    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();

    optix::Context context = optix::Context::create();

    OScintillatorLib* oscin ;  
    oscin = new OScintillatorLib(context, slib );

    oscin->convert();


    return 0 ; 
}
