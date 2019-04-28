#include <cassert>
#include <string>
#include <sstream>

#include "BTimes.hh"
#include "NPY.hpp"

#include "OConfig.hh"
#include "OContext.hh"
#include "Opticks.hh"
#include "OpticksBufferControl.hh"

#include "OPTICKS_LOG.hh"

struct Evt 
{
   Evt(unsigned size_);
   void check();
   std::string description();

   unsigned size ; 
   BTimes* times ; 
   NPY<float>* genstep ;
   NPY<float>* photon  ;

};


Evt::Evt(unsigned size_) 
   :
   size(size_),
   times(new BTimes),  
   genstep(NPY<float>::make(size,1,4)),  
   photon(NPY<float>::make(size,1,4))  
{

   // simplifications of 
   //    OpticksEvent::createSpec
   //    OPropagator::initEventBuffers

   const char* genstep_ctrl = "OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY" ;
   const char* photon_ctrl  = "OPTIX_INPUT_OUTPUT,INTEROP_PTR_FROM_OPENGL" ;
 
   OpticksBufferControl c_genstep(genstep->getBufferControlPtr()); 
   c_genstep.add(genstep_ctrl); 

   OpticksBufferControl c_photon(photon->getBufferControlPtr()); 
   c_photon.add(photon_ctrl);
   c_photon.add(OpticksBufferControl::COMPUTE_MODE_);  // needed otherwise download skipped

   genstep->fill(float(size));
   //genstep->dump();

   photon->zero();
}


void Evt::check()
{
    //photon->dump();
    bool dump = false ;  
    float maxdiff = genstep->maxdiff(photon, dump);
    LOG(info) << " size " << size << " maxdiff " << maxdiff  ;
    assert(maxdiff < 1e-6 ); 
}


std::string Evt::description()
{
    std::stringstream ss ; 
    ss << "evt:" 
        << size  
        ;
    return ss.str();
}


/**
bufferTest
============

Test:

* doing expensive validation, compilation and prelaunch 
  at initialization, by using zero sized buffers as standins 
  for real event buffers

* changing buffer sizes between "hot" launches 


**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    Opticks ok(argc, argv, "--compute");
    ok.configure();

    unsigned version = OConfig::OptiXVersion()  ;
    LOG(info) << argv[0] << " OPTIX_VERSION " << version ; 
    //bool with_top = OConfig::DefaultWithTop() ;  // must set false with 3080, seemingly doesnt matter with 40000

    OContext* ctx = OContext::Create(&ok);
    optix::Context context = ctx->getContext();
    int entry = ctx->addEntry("bufferTest.cu", "bufferTest", "exception");

    // using zero sized buffers allows to prelaunch in initialization
    // so once have real events can just do the much faster launch 
 
    Evt* evt0 = new Evt(0) ;

    optix::Buffer m_genstep_buffer = ctx->createBuffer<float>( evt0->genstep, "genstep");
    context["genstep_buffer"]->set( m_genstep_buffer );

    optix::Buffer m_photon_buffer = ctx->createBuffer<float>( evt0->photon, "photon");
    context["photon_buffer"]->set( m_photon_buffer );

    ctx->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  entry,  0, 0, evt0->times);

    LOG(info) <<  evt0->description() ;


    for(unsigned i=0 ; i < 10 ; i++)
    {
         unsigned size = 100+i*100 ; 
   
         Evt* evt = new Evt(size) ;

         m_genstep_buffer->setSize(evt->size);
         m_photon_buffer->setSize(evt->size);

         OContext::upload<float>(m_genstep_buffer, evt->genstep);  // compute mode style

         ctx->launch( OContext::LAUNCH, entry,  evt->size, 1, evt->times);

         OContext::download<float>( m_photon_buffer, evt->photon );

         evt->check();

         LOG(info) <<  evt->description() ;
    }

    delete ctx ; 

    return 0 ;     
}

/*

delta:optixrap blyth$ bufferTest 
2016-09-16 19:42:19.236 FATAL [8117] [OpticksProfile::stamp@87] OpticksProfile::stamp OpticksRun::OpticksRun_0 (0,42139.2,0,2583)
2016-09-16 19:42:19.236 FATAL [8117] [OpticksProfile::stamp@87] OpticksProfile::stamp Opticks::Opticks_0 (0.00390625,0.00390625,10,10)
2016-09-16 19:42:19.237 INFO  [8117] [main@107] bufferTest OPTIX_VERSION 3080
2016-09-16 19:42:19.724 INFO  [8117] [OContext::close@209] OContext::close numEntryPoint 1
2016-09-16 19:42:19.733 INFO  [8117] [OContext::launch@235] OContext::launch entry 0 width 0 height 0
libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Invalid context (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Validation error: Node validation failed for 'top_object':
Validation error: Group does not have an Acceleration Structure, [4915492], [4915305])
Abort trap: 6


*/



