#include "Opticks.hh"

#include "OptiXTest.hh"
#include "OContext.hh"

#include "NPY.hpp"
#include "NLoad.hpp"

#include "OBuf.hh"

#include "BOpticksEvent.hh"
#include "BOpticksResource.hh"

#include "OpticksEvent.hh"
#include "OpticksBufferControl.hh"
#include "BOpticksResource.hh"

#include "OXRAP_LOG.hh"
#include "PLOG.hh"


int main( int argc, char** argv ) 
{
    PLOG_(argc, argv);
    OXRAP_LOG__ ; 

    Opticks* ok = new Opticks(argc, argv, "--compute" );
    ok->configure();


    NPY<float>* npy = NLoad::Gensteps("juno", "cerenkov", "1") ; 
    assert(npy);
    npy->dump("NPY::dump::before", 2);

    // manual buffer control, normally done via spec in okc-/OpticksEvent 
    npy->setBufferControl(OpticksBufferControl::Parse("OPTIX_INPUT_OUTPUT"));

    optix::Context context = optix::Context::create();

    //OContext::Mode_t mode = OContext::COMPUTE ;

    OContext* m_ocontext = new OContext(context, ok, false );

    unsigned entry = m_ocontext->addEntry("LTminimalTest.cu", "minimal", "exception");


    optix::Buffer buffer = m_ocontext->createBuffer<float>( npy, "demo");
    context["output_buffer"]->set(buffer);
    OBuf* genstep_buf = new OBuf("genstep", buffer);

    OContext::upload(buffer, npy);

    genstep_buf->dump<unsigned int>("LT::OBuf test: ", 6*4, 3, 6*4*10);
    LOG(info) << "check OBuf begin.";
    // LT: check OBuf
    npy->zero();
    genstep_buf->download(npy);
    npy->dump("NPY::dump::after", 2);
    LOG(info) << "check OBuf end.";

    unsigned ni = 10 ; 
    m_ocontext->launch( OContext::VALIDATE,  entry, ni, 1);
    genstep_buf->dump<unsigned int>("LT::OBuf test after VALIDATE: ", 6*4, 3, 6*4*10);
    m_ocontext->launch( OContext::COMPILE,   entry, ni, 1);
    genstep_buf->dump<unsigned int>("LT::OBuf test after COMPILE: ", 6*4, 3, 6*4*10);
    m_ocontext->launch( OContext::PRELAUNCH, entry, ni, 1);
    genstep_buf->dump<unsigned int>("LT::OBuf test after PRELAUNCH: ", 6*4, 3, 6*4*10);
    m_ocontext->launch( OContext::LAUNCH,    entry, ni, 1);
    genstep_buf->dump<unsigned int>("LT::OBuf test after LAUNCH: ", 6*4, 3, 6*4*10);

    npy->zero();

    OContext::download( buffer, npy );

    NPYBase::setGlobalVerbose();

    // npy->dump();
    npy->save("$TMP/OOContextUploadDownloadTest_1.npy");

    return 0;
}
