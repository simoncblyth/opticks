#include <string>
#include <sstream>

#include "NPY.hpp"

#include "OContext.hh"
#include "OTimes.hh"
#include "Opticks.hh"
#include "OpticksBufferControl.hh"

#include "OKCORE_LOG.hh"
#include "OXRAP_LOG.hh"

#include "PLOG.hh"


struct Evt 
{
   Evt(unsigned size_);
   void check();
   std::string description();

   unsigned size ; 
   OTimes* times ; 
   NPY<float>* genstep ;
   NPY<float>* photon  ;

};


Evt::Evt(unsigned size_) 
   :
   size(size_),
   times(new OTimes),  
   genstep(NPY<float>::make(size,1,4)),  
   photon(NPY<float>::make(size,1,4))  
{

   // simplifications of 
   //    OpticksEvent::createSpec
   //    OPropagator::initEventBuffers

   const char* genstep_ctrl = "OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY" ;
   const char* photon_ctrl  = "OPTIX_INPUT_OUTPUT,PTR_FROM_OPENGL" ;
 
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
        << times->description(" ") 
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
    PLOG_(argc, argv);    

    OKCORE_LOG__ ; 
    OXRAP_LOG__ ; 

    Opticks ok(argc, argv);
    ok.configure();

    LOG(info) << argv[0] ; 

    optix::Context context = optix::Context::create();
    OContext ctx(context, OContext::COMPUTE);
    int entry = ctx.addEntry("bufferTest.cu.ptx", "bufferTest", "exception");

    // using zero sized buffers allows to prelaunch in initialization
    // so once have real events can just do the much faster launch 
 
    Evt* evt0 = new Evt(0) ;

    optix::Buffer m_genstep_buffer = ctx.createBuffer<float>( evt0->genstep, "genstep");
    context["genstep_buffer"]->set( m_genstep_buffer );

    optix::Buffer m_photon_buffer = ctx.createBuffer<float>( evt0->photon, "photon");
    context["photon_buffer"]->set( m_photon_buffer );

    ctx.launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  entry,  0, 0, evt0->times);

    LOG(info) <<  evt0->description() ;


    for(unsigned i=0 ; i < 10 ; i++)
    {
         unsigned size = 100+i*100 ; 
   
         Evt* evt = new Evt(size) ;

         //context["genstep_buffer"]->getBuffer()->setSize(evt->size);
         //context["photon_buffer"]->getBuffer()->setSize(evt->size);

         m_genstep_buffer->setSize(evt->size);
         m_photon_buffer->setSize(evt->size);

         OContext::upload<float>(m_genstep_buffer, evt->genstep);  // compute mode style

         ctx.launch( OContext::LAUNCH, entry,  evt->size, 1, evt->times);

         OContext::download<float>( m_photon_buffer, evt->photon );

         evt->check();

         LOG(info) <<  evt->description() ;
    }
    return 0 ;     
}

