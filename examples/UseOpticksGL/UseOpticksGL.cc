// opticksgl/tests/OOAxisAppCheck.cc
// npy-
#include "NPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

// oglrap-
#include "AxisApp.hh"

// optixrap-
#include "Opticks.hh"
#include "OContext.hh"

// opticksgl-
#include "OAxisTest.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    Opticks ok(argc, argv, "--interop --renderlooplimit 2000");
    ok.configure();

    LOG(info) << argv[0] ; 

    AxisApp axa(&ok); 
    NPY<float>* npy = axa.getAxisData();
    assert(npy->hasShape(3,3,4));

    /*
    MultiViewNPY* mvn = aa.getAxisAttr();
    ViewNPY* vpos = (*mvn)["vpos"];
    NPYBase* npyb = vpos->getNPY();  // NB same npy holds vpos, vdir, vcol
    assert(npy == npyb);
    */
   
    //OContext::Mode_t mode = OContext::INTEROP ;

    const char* cmake_target = "OptiXRap" ; 
    const char* ptxrel = NULL ; 

    OContext* ocontext = OContext::Create(&ok, cmake_target, ptxrel );

    OAxisTest* oat = new OAxisTest(ocontext, npy);
    oat->prelaunch();

    axa.setLauncher(oat);
    axa.renderLoop();

    return 0 ; 
}

// note flakiness, sometimes the axis appears sometimes not


