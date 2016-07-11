
// npy-
#include "NPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

// oglrap-
#include "AxisApp.hh"

// optixrap-
#include "OContext.hh"

// opticksgl-
#include "OAxisTest.hh"

#include "OGLRAP_LOG.hh"
#include "OXRAP_LOG.hh"
#include "OKGL_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OGLRAP_LOG__ ; 
    OXRAP_LOG__ ; 
    OKGL_LOG__ ; 

    LOG(info) << argv[0] ; 

    AxisApp axa(argc, argv); 
    NPY<float>* npy = axa.getAxisData();
    assert(npy->hasShape(3,3,4));

    /*
    MultiViewNPY* mvn = aa.getAxisAttr();
    ViewNPY* vpos = (*mvn)["vpos"];
    NPYBase* npyb = vpos->getNPY();  // NB same npy holds vpos, vdir, vcol
    assert(npy == npyb);
    */
   
    OContext::Mode_t mode = OContext::INTEROP ;
    optix::Context context = optix::Context::create();

    OContext* m_ocontext = new OContext(context, mode, false );

    OAxisTest* oat = new OAxisTest(m_ocontext, npy);
    oat->prelaunch();

    axa.setLauncher(oat);
    axa.renderLoop();

    return 0 ; 
}

// note flakiness, sometimes the axis appears sometimes not


