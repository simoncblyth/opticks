
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

    const char* slice = "0:1" ; 
    oscin->convert(slice);


    // Initially suspected below flaky FAIL may be due to executable termination 
    // whilst GPU still active (see thrustrap-/tests/TBufTest.cu ).
    //
    // But now think this the error was almost certainly this due to two scintillators being 
    // squeezed into a buffer for one, this caused a Linux porting issue
    // previously that was fixed with the slicing to pick the first. 
    //

    LOG(info) << "DONE"  ;

    return 0 ; 
}

/*

Have observered opticks-t CTest infrequent flakiness with 
this test, it occasionally fails with below error
but on rerunning it does not fail...::

    The following tests FAILED:
        142 - OptiXRapTest.OScintillatorLibTest (OTHER_FAULT)
    Errors while running CTest

Recording the logs of test running, does not reveal
the cause.


*/
