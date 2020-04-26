/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// optixrap/tests/bufferTest.cc

#include <cassert>
#include <string>
#include <sstream>

#include "NPY.hpp"

#include "OConfig.hh"
#include "OContext.hh"
#include "BTimes.hh"
#include "SSys.hh"
#include "Opticks.hh"
#include "OpticksBufferControl.hh"

#include "OPTICKS_LOG.hh"


struct Evt 
{
   Evt(unsigned size_);
   void check();
   std::string description();
   std::string brief();

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

   //const char* genstep_ctrl = "OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY" ;
   //const char* photon_ctrl  = "OPTIX_INPUT_OUTPUT,INTEROP_PTR_FROM_OPENGL" ;

   const char* genstep_ctrl = "OPTIX_INPUT_ONLY" ;
   const char* photon_ctrl  = "OPTIX_OUTPUT_ONLY" ;
 
 
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


std::string Evt::brief()
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

Simply copies float4 values from the "genstep" buffer
into the "photon" buffer.

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


    OContext* ctx = OContext::Create(&ok );
    optix::Context context = ctx->getContext(); 

    context->setPrintEnabled(true); 


    //const char* progname = "bufferTest" ; 
    //const char* progname = "bufferTest_0" ; 

    const char* ekey = "USEOPTIXRAP_PROGNAME" ;
    const char* edef = "bufferTest" ; 
    const char* progname = SSys::getenvvar(ekey, edef) ; 

    int entry = ctx->addEntry("bufferTest.cu", progname, "exception");





    // using zero sized buffers allows to prelaunch in initialization
    // so once have real events can just do the much faster launch 
 
    Evt* evt0 = new Evt(0) ;

    optix::Buffer m_genstep_buffer = ctx->createBuffer<float>( evt0->genstep, "gensteps");
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

         LOG(info) <<  evt->brief() ;
    }

    std::cout << std::endl ; 
    std::cout << ekey << "=" << edef << " " << argv[0] << " ## to modify the program funcname launched " << std::endl ; 
    std::cout << std::endl ; 
    std::cout << "also use VERBOSE=1 envvar to see the oxrap buffer setup " << std::endl ; 

    delete ctx ; 
  

    return 0 ;     
}


